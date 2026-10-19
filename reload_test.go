package do

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// ---------- 测试用服务类型 ----------

// reloadTestDB 模拟数据库连接：可记录世代、关停状态、健康检查行为。
type reloadTestDB struct {
	gen         int
	healthErr   error
	healthDelay time.Duration

	closeMu sync.Mutex
	closed  int32
}

func (d *reloadTestDB) HealthCheck(ctx context.Context) error {
	if d.healthDelay > 0 {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(d.healthDelay):
		}
	}

	return d.healthErr
}

func (d *reloadTestDB) Shutdown(_ context.Context) error {
	d.closeMu.Lock()
	defer d.closeMu.Unlock()
	d.closed++
	return nil
}

func (d *reloadTestDB) closeCount() int32 {
	d.closeMu.Lock()
	defer d.closeMu.Unlock()
	return d.closed
}

func (d *reloadTestDB) isClosed() bool {
	return d.closeCount() > 0
}

// reloadTestHandler 模拟依赖数据库的处理器，构造时注入 DB 世代。
type reloadTestHandler struct {
	gen   int
	dbGen int
	db    *reloadTestDB
}

func (h *reloadTestHandler) Shutdown(_ context.Context) error { return nil }

type reloadTestShutdowner struct {
	id    string
	order *[]string
	mu    *sync.Mutex
}

func (s *reloadTestShutdowner) Shutdown(_ context.Context) error {
	s.mu.Lock()
	*s.order = append(*s.order, s.id)
	s.mu.Unlock()
	return nil
}

// reloadTestStore 是 DB 暴露的接口，用于验证 alias 在目标重载后的行为。
type reloadTestStore interface {
	StoreGen() int
}

func (d *reloadTestDB) StoreGen() int { return d.gen }

// reloadTestPair 用于并发场景：一次解析中读到的两个依赖必须同世代。
type reloadTestPair struct {
	aGen int32
	bGen int32
}

func newReloadTestRoot(t *testing.T) *RootScope {
	t.Helper()
	root := New()
	t.Cleanup(func() {
		root.Shutdown()
	})
	return root
}

func provideReloadDB(i Injector, name string, db *reloadTestDB) {
	ProvideNamed(i, name, func(_ Injector) (*reloadTestDB, error) {
		return db, nil
	})
}

// ---------- 测试用例 ----------

// 成功路径：成批替换 DB + Handler，旧实例按序关闭，未受影响实例保持原样。
func TestRootScope_ReloadWithContext_Success(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)

	db1 := &reloadTestDB{gen: 1}
	handler1 := &reloadTestHandler{gen: 10}
	keepOrder := []string{}
	keep := &reloadTestShutdowner{id: "keep", order: &keepOrder, mu: &sync.Mutex{}}

	provideReloadDB(root, "db", db1)
	ProvideNamed(root, "handler", func(i Injector) (*reloadTestHandler, error) {
		db := MustInvokeNamed[*reloadTestDB](i, "db")
		return &reloadTestHandler{gen: handler1.gen, dbGen: db.gen, db: db}, nil
	})
	ProvideNamedValue(root, "keep", keep)

	// 重载前全部调用过
	oldDB := MustInvokeNamed[*reloadTestDB](root, "db")
	oldHandler := MustInvokeNamed[*reloadTestHandler](root, "handler")
	oldKeep := MustInvokeNamed[*reloadTestShutdowner](root, "keep")
	is.Equal(1, oldDB.gen)
	is.Equal(1, oldHandler.dbGen)

	db2 := &reloadTestDB{gen: 2}
	err := root.ReloadWithContext(context.Background(), func(i Injector) error {
		OverrideNamed(i, "db", func(_ Injector) (*reloadTestDB, error) {
			return db2, nil
		})
		OverrideNamed(i, "handler", func(i Injector) (*reloadTestHandler, error) {
			db := MustInvokeNamed[*reloadTestDB](i, "db")
			return &reloadTestHandler{gen: 20, dbGen: db.gen, db: db}, nil
		})
		return nil
	})
	is.NoError(err)

	// 后续 Invoke 必须读到新一轮配置
	newDB := MustInvokeNamed[*reloadTestDB](root, "db")
	newHandler := MustInvokeNamed[*reloadTestHandler](root, "handler")
	is.Equal(2, newDB.gen)
	is.Equal(20, newHandler.gen)
	is.Equal(2, newHandler.dbGen)
	is.Same(newDB, newHandler.db)

	// 旧实例已关闭
	is.True(db1.isClosed())
	is.NotSame(oldDB, newDB)
	is.NotSame(oldHandler, newHandler)

	// 未受影响实例保持原样且未被关闭
	stillKeep := MustInvokeNamed[*reloadTestShutdowner](root, "keep")
	is.Same(oldKeep, stillKeep)
}

// 子作用域在重载前调用过的根服务，重载后同样解析到新一轮实例。
func TestRootScope_ReloadWithContext_ChildScopeSeesNewRound(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)
	child := root.Scope("child")

	db1 := &reloadTestDB{gen: 1}
	provideReloadDB(root, "db", db1)

	fromChild := MustInvokeNamed[*reloadTestDB](child, "db")
	is.Equal(1, fromChild.gen)

	db2 := &reloadTestDB{gen: 2}
	err := root.ReloadWithContext(context.Background(), func(i Injector) error {
		OverrideNamed(i, "db", func(_ Injector) (*reloadTestDB, error) {
			return db2, nil
		})
		return nil
	})
	is.NoError(err)

	againFromChild := MustInvokeNamed[*reloadTestDB](child, "db")
	is.Equal(2, againFromChild.gen)
	is.Same(db2, againFromChild)
	is.True(db1.isClosed())
}

// configure 返回错误：线上注册与实例完全不受影响，并可立即重试成功。
func TestRootScope_ReloadWithContext_ConfigureError(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)
	db1 := &reloadTestDB{gen: 1}
	provideReloadDB(root, "db", db1)
	old := MustInvokeNamed[*reloadTestDB](root, "db")

	reloadErr := errors.New("bad config")
	err := root.ReloadWithContext(context.Background(), func(i Injector) error {
		OverrideNamed(i, "db", func(_ Injector) (*reloadTestDB, error) {
			return &reloadTestDB{gen: 2}, nil
		})
		return reloadErr
	})
	is.ErrorIs(err, reloadErr)

	stillOld := MustInvokeNamed[*reloadTestDB](root, "db")
	is.Same(old, stillOld)
	is.False(db1.isClosed())

	// 允许再次重试
	db3 := &reloadTestDB{gen: 3}
	err = root.ReloadWithContext(context.Background(), func(i Injector) error {
		OverrideNamed(i, "db", func(_ Injector) (*reloadTestDB, error) {
			return db3, nil
		})
		return nil
	})
	is.NoError(err)
	is.Same(db3, MustInvokeNamed[*reloadTestDB](root, "db"))
	is.True(db1.isClosed())
}

// configure panic 同样不能破坏现有服务。
func TestRootScope_ReloadWithContext_ConfigurePanic(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)
	db1 := &reloadTestDB{gen: 1}
	provideReloadDB(root, "db", db1)
	MustInvokeNamed[*reloadTestDB](root, "db")

	err := root.ReloadWithContext(context.Background(), func(_ Injector) error {
		panic("boom")
	})
	is.ErrorContains(err, "boom")

	is.Equal(db1, MustInvokeNamed[*reloadTestDB](root, "db"))
	is.False(db1.isClosed())
}

// configure 中重复 Provide 已注册服务会 panic，应被转成错误且旧服务无损。
func TestRootScope_ReloadWithContext_DuplicateProvideRejected(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)
	db1 := &reloadTestDB{gen: 1}
	provideReloadDB(root, "db", db1)

	err := root.ReloadWithContext(context.Background(), func(i Injector) error {
		ProvideNamed(i, "db", func(_ Injector) (*reloadTestDB, error) {
			return &reloadTestDB{gen: 2}, nil
		})
		return nil
	})
	is.ErrorContains(err, "already been declared")
	is.Same(db1, MustInvokeNamed[*reloadTestDB](root, "db"))
}

// 同名多次 Override 以最后一次为准。
func TestRootScope_ReloadWithContext_LastOverrideWins(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)
	provideReloadDB(root, "db", &reloadTestDB{gen: 1})
	MustInvokeNamed[*reloadTestDB](root, "db")

	dbFinal := &reloadTestDB{gen: 9}
	err := root.ReloadWithContext(context.Background(), func(i Injector) error {
		OverrideNamed(i, "db", func(_ Injector) (*reloadTestDB, error) {
			return &reloadTestDB{gen: 2}, nil
		})
		OverrideNamed(i, "db", func(_ Injector) (*reloadTestDB, error) {
			return dbFinal, nil
		})
		return nil
	})
	is.NoError(err)
	is.Equal(9, MustInvokeNamed[*reloadTestDB](root, "db").gen)
}

// 新服务构造失败：已安装的新包装整体回滚，旧实例继续可用，新实例被清理。
func TestRootScope_ReloadWithContext_BuildError(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)
	db1 := &reloadTestDB{gen: 1}
	provideReloadDB(root, "db", db1)
	old := MustInvokeNamed[*reloadTestDB](root, "db")

	buildErr := errors.New("cannot dial database")
	err := root.ReloadWithContext(context.Background(), func(i Injector) error {
		OverrideNamed(i, "db", func(_ Injector) (*reloadTestDB, error) {
			return nil, buildErr
		})
		return nil
	})
	is.ErrorIs(err, buildErr)

	// 旧注册与旧实例继续可用
	is.Same(old, MustInvokeNamed[*reloadTestDB](root, "db"))
	is.False(db1.isClosed())

	// 失败后可重试成功
	db2 := &reloadTestDB{gen: 2}
	err = root.ReloadWithContext(context.Background(), func(i Injector) error {
		OverrideNamed(i, "db", func(_ Injector) (*reloadTestDB, error) {
			return db2, nil
		})
		return nil
	})
	is.NoError(err)
	is.Same(db2, MustInvokeNamed[*reloadTestDB](root, "db"))
}

// 健康检查失败：新实例被关闭，旧实例继续提供服务。
func TestRootScope_ReloadWithContext_HealthCheckError(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)
	db1 := &reloadTestDB{gen: 1}
	provideReloadDB(root, "db", db1)
	old := MustInvokeNamed[*reloadTestDB](root, "db")

	db2 := &reloadTestDB{gen: 2, healthErr: errors.New("ping failed")}
	err := root.ReloadWithContext(context.Background(), func(i Injector) error {
		OverrideNamed(i, "db", func(_ Injector) (*reloadTestDB, error) {
			return db2, nil
		})
		return nil
	})
	is.ErrorContains(err, "health check failed")

	// 旧实例可用，新实例已清理
	is.Same(old, MustInvokeNamed[*reloadTestDB](root, "db"))
	is.True(db2.isClosed())
	is.False(db1.isClosed())
}

// 上下文在健康检查阶段取消：回滚到旧轮，新实例清理，旧实例继续可用。
func TestRootScope_ReloadWithContext_ContextCanceled(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)
	db1 := &reloadTestDB{gen: 1}
	provideReloadDB(root, "db", db1)
	old := MustInvokeNamed[*reloadTestDB](root, "db")

	db2 := &reloadTestDB{gen: 2, healthDelay: time.Hour}
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	err := root.ReloadWithContext(ctx, func(i Injector) error {
		OverrideNamed(i, "db", func(_ Injector) (*reloadTestDB, error) {
			return db2, nil
		})
		return nil
	})
	is.Error(err)
	is.True(errors.Is(err, context.DeadlineExceeded))

	// db2 的健康检查尊重 ctx，会很快退出并被回滚关闭
	is.Eventually(func() bool { return db2.isClosed() }, 2*time.Second, 10*time.Millisecond)
	is.Same(old, MustInvokeNamed[*reloadTestDB](root, "db"))
	is.False(db1.isClosed())
}

// 启动前 ctx 已取消：直接失败，configure 不应被调用。
func TestRootScope_ReloadWithContext_AlreadyCanceled(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)
	provideReloadDB(root, "db", &reloadTestDB{gen: 1})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	called := false
	err := root.ReloadWithContext(ctx, func(_ Injector) error {
		called = true
		return nil
	})
	is.Error(err)
	is.False(called)
}

// configure 内调用 Invoke 不会自锁；失败后旧配置仍然完好。
func TestRootScope_ReloadWithContext_InvokeInsideConfigure(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)
	db1 := &reloadTestDB{gen: 1}
	provideReloadDB(root, "db", db1)
	MustInvokeNamed[*reloadTestDB](root, "db")

	// configure 中读取当前实例（拿旧世代做计算），随后返回错误：不得死锁。
	err := root.ReloadWithContext(context.Background(), func(i Injector) error {
		current := MustInvokeNamed[*reloadTestDB](i, "db")
		is.Equal(1, current.gen)
		return errors.New("abort after read")
	})
	is.ErrorContains(err, "abort after read")
	is.Same(db1, MustInvokeNamed[*reloadTestDB](root, "db"))

	// 成功轮同样支持在 configure 中读取
	db2 := &reloadTestDB{gen: 2}
	err = root.ReloadWithContext(context.Background(), func(i Injector) error {
		current := MustInvokeNamed[*reloadTestDB](i, "db")
		is.Equal(1, current.gen)
		OverrideNamed(i, "db", func(_ Injector) (*reloadTestDB, error) {
			return db2, nil
		})
		return nil
	})
	is.NoError(err)
	is.Same(db2, MustInvokeNamed[*reloadTestDB](root, "db"))
}

// nil configure 直接报错。
func TestRootScope_ReloadWithContext_NilConfigure(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)
	is.Error(root.ReloadWithContext(context.Background(), nil))
}

// 回滚场景：本轮新增（旧轮不存在）的服务必须从注册中移除。
func TestRootScope_ReloadWithContext_NewNameRolledBack(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)
	provideReloadDB(root, "db", &reloadTestDB{gen: 1})
	MustInvokeNamed[*reloadTestDB](root, "db")

	err := root.ReloadWithContext(context.Background(), func(i Injector) error {
		OverrideNamed(i, "db", func(_ Injector) (*reloadTestDB, error) {
			return &reloadTestDB{gen: 2, healthErr: errors.New("nope")}, nil
		})
		// 顺带新增一个旧轮没有的服务
		OverrideNamed(i, "brand-new", func(_ Injector) (*reloadTestHandler, error) {
			return &reloadTestHandler{gen: 1}, nil
		})
		return nil
	})
	is.Error(err)

	is.False(root.serviceExist("brand-new"))
	is.True(root.serviceExist("db"))
	is.Equal(1, MustInvokeNamed[*reloadTestDB](root, "db").gen)

	// 重试成功后新名字可以正常注册
	err = root.ReloadWithContext(context.Background(), func(i Injector) error {
		OverrideNamed(i, "brand-new", func(_ Injector) (*reloadTestHandler, error) {
			return &reloadTestHandler{gen: 2}, nil
		})
		return nil
	})
	is.NoError(err)
	is.True(root.serviceExist("brand-new"))
}

// 未调用过的 lazy 被替换后：旧实例无需关闭；新一轮首次 Invoke 使用新构造器。
func TestRootScope_ReloadWithContext_NeverInvokedLazy(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)
	db1 := &reloadTestDB{gen: 1}
	provideReloadDB(root, "db", db1)
	// 注意：不 Invoke

	db2 := &reloadTestDB{gen: 2}
	err := root.ReloadWithContext(context.Background(), func(i Injector) error {
		OverrideNamed(i, "db", func(_ Injector) (*reloadTestDB, error) {
			return db2, nil
		})
		return nil
	})
	is.NoError(err)

	// 重载校验会预先构造新服务；旧实例从未构造，不应被关闭
	is.False(db1.isClosed())
	is.Same(db2, MustInvokeNamed[*reloadTestDB](root, "db"))
}

// eager 值替换：旧值在提交后关闭；健康检查失败时旧值保持可用。
func TestRootScope_ReloadWithContext_EagerValue(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)
	cfg1 := &reloadTestDB{gen: 1}
	ProvideNamedValue(root, "cfg", cfg1)
	is.Same(cfg1, MustInvokeNamed[*reloadTestDB](root, "cfg"))

	cfg2 := &reloadTestDB{gen: 2}
	err := root.ReloadWithContext(context.Background(), func(i Injector) error {
		OverrideNamedValue(i, "cfg", cfg2)
		return nil
	})
	is.NoError(err)
	is.Same(cfg2, MustInvokeNamed[*reloadTestDB](root, "cfg"))
	is.True(cfg1.isClosed())

	// 失败回滚：旧 eager 值继续可用
	cfg3Bad := &reloadTestDB{gen: 3, healthErr: errors.New("bad")}
	err = root.ReloadWithContext(context.Background(), func(i Injector) error {
		OverrideNamedValue(i, "cfg", cfg3Bad)
		return nil
	})
	is.Error(err)
	is.Same(cfg2, MustInvokeNamed[*reloadTestDB](root, "cfg"))
	is.True(cfg3Bad.isClosed())
}

// transient 工厂：重载会试构造验证；试构造实例被关闭，后续 Invoke 拿到新实例；
// 工厂失败则整轮回滚。
func TestRootScope_ReloadWithContext_Transient(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)

	var builds int32
	ProvideNamedTransient(root, "t-db", func(_ Injector) (*reloadTestDB, error) {
		n := atomic.AddInt32(&builds, 1)
		return &reloadTestDB{gen: int(n)}, nil
	})

	first := MustInvokeNamed[*reloadTestDB](root, "t-db")
	is.Equal(1, first.gen)

	err := root.ReloadWithContext(context.Background(), func(i Injector) error {
		OverrideNamedTransient(i, "t-db", func(_ Injector) (*reloadTestDB, error) {
			n := atomic.AddInt32(&builds, 1)
			return &reloadTestDB{gen: 100 + int(n)}, nil
		})
		return nil
	})
	is.NoError(err)

	// 试构造产生了一个实例（gen 102），后续调用拿到的是 103
	second, err := InvokeNamed[*reloadTestDB](root, "t-db")
	is.NoError(err)
	is.Equal(103, second.gen)

	// 工厂失败必须回滚
	err = root.ReloadWithContext(context.Background(), func(i Injector) error {
		OverrideNamedTransient(i, "t-db", func(_ Injector) (*reloadTestDB, error) {
			return nil, errors.New("factory down")
		})
		return nil
	})
	is.ErrorContains(err, "factory down")

	third, err := InvokeNamed[*reloadTestDB](root, "t-db")
	is.NoError(err)
	is.Equal(104, third.gen)
}

// 旧实例关停顺序遵循 DAG：依赖方（handler）先于被依赖方（db）关闭。
func TestRootScope_ReloadWithContext_OldInstancesShutdownInDAGOrder(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)

	var mu sync.Mutex
	order := []string{}
	mkShutdowner := func(id string) *reloadTestShutdowner {
		return &reloadTestShutdowner{id: id, order: &order, mu: &mu}
	}

	db1 := mkShutdowner("db")
	handler1 := mkShutdowner("handler")

	ProvideNamed(root, "db", func(_ Injector) (*reloadTestShutdowner, error) { return db1, nil })
	ProvideNamed(root, "handler", func(i Injector) (*reloadTestShutdowner, error) {
		MustInvokeNamed[*reloadTestShutdowner](i, "db") // 建立 handler -> db 依赖边
		return handler1, nil
	})
	MustInvokeNamed[*reloadTestShutdowner](root, "db")
	MustInvokeNamed[*reloadTestShutdowner](root, "handler")

	err := root.ReloadWithContext(context.Background(), func(i Injector) error {
		OverrideNamed(i, "db", func(_ Injector) (*reloadTestShutdowner, error) {
			return mkShutdowner("db-new"), nil
		})
		OverrideNamed(i, "handler", func(i Injector) (*reloadTestShutdowner, error) {
			MustInvokeNamed[*reloadTestShutdowner](i, "db")
			return mkShutdowner("handler-new"), nil
		})
		return nil
	})
	is.NoError(err)

	is.Equal([]string{"handler", "db"}, order)
}

// alias 场景：只重载目标服务，alias 解析自动指向新目标；
// 旧 alias 包装器不能被关停（否则会误关停共享目标），旧目标仅关闭一次。
func TestRootScope_ReloadWithContext_AliasFollowsNewTarget(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)
	db1 := &reloadTestDB{gen: 1}
	provideReloadDB(root, "db", db1)
	is.NoError(AsNamed[*reloadTestDB, reloadTestStore](root, "db", "store"))

	store1 := MustInvokeNamed[reloadTestStore](root, "store")
	is.Equal(1, store1.StoreGen())

	db2 := &reloadTestDB{gen: 2}
	err := root.ReloadWithContext(context.Background(), func(i Injector) error {
		OverrideNamed(i, "db", func(_ Injector) (*reloadTestDB, error) {
			return db2, nil
		})
		return nil
	})
	is.NoError(err)

	store2 := MustInvokeNamed[reloadTestStore](root, "store")
	is.Equal(2, store2.StoreGen())
	is.Equal(int32(1), db1.closeCount()) // 旧目标仅关闭一次
}

// 重载与并发 Invoke 交错：一次解析中读到的两个依赖必须是同一世代，
// 不允许出现 (1,2) 这类跨轮混搭。
func TestRootScope_ReloadWithContext_ConcurrentInvokesDoNotMixRounds(t *testing.T) {
	testWithTimeout(t, 30*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)

	var genCounter int32
	atomic.StoreInt32(&genCounter, 0)

	provideNamedGen := func(i Injector, name string) {
		ProvideNamed(i, name, func(_ Injector) (*reloadTestDB, error) {
			return &reloadTestDB{gen: int(atomic.LoadInt32(&genCounter))}, nil
		})
	}
	overrideNamedGen := func(i Injector, name string, gen int32) {
		OverrideNamed(i, name, func(_ Injector) (*reloadTestDB, error) {
			return &reloadTestDB{gen: int(gen)}, nil
		})
	}

	atomic.StoreInt32(&genCounter, 1)
	provideNamedGen(root, "dep-a")
	provideNamedGen(root, "dep-b")

	// pair 使用 transient：每次解析都重新读取 a、b，
	// 并在两次读取之间让出时间片，最大化与重载提交交错的概率。
	ProvideNamedTransient(root, "pair", func(i Injector) (*reloadTestPair, error) {
		a := MustInvokeNamed[*reloadTestDB](i, "dep-a")
		time.Sleep(time.Millisecond)
		b := MustInvokeNamed[*reloadTestDB](i, "dep-b")
		if a.gen != b.gen {
			return nil, fmt.Errorf("mixed rounds: a=%d b=%d", a.gen, b.gen)
		}
		return &reloadTestPair{aGen: int32(a.gen), bGen: int32(b.gen)}, nil
	})

	stop := make(chan struct{})
	var wg sync.WaitGroup
	var resolveErr atomic.Value

	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				if _, err := InvokeNamed[*reloadTestPair](root, "pair"); err != nil {
					resolveErr.Store(err)
					return
				}
			}
		}()
	}

	// 连续进行多轮重载，替换两个依赖
	for round := int32(2); round <= 6; round++ {
		err := root.ReloadWithContext(context.Background(), func(i Injector) error {
			overrideNamedGen(i, "dep-a", round)
			overrideNamedGen(i, "dep-b", round)
			return nil
		})
		is.NoError(err)
	}

	close(stop)
	wg.Wait()

	if e := resolveErr.Load(); e != nil {
		t.Fatalf("concurrent resolution mixed rounds: %v", e)
	}

	// 最终一轮所有解析读到世代 6
	pair := MustInvokeNamed[*reloadTestPair](root, "pair")
	is.Equal(int32(6), pair.aGen)
	is.Equal(int32(6), pair.bGen)
}

// 重载进行中，外部 Invoke 在门禁处等待；重载结束后拿到新实例，不会死锁。
func TestRootScope_ReloadWithContext_InvokeWaitsForCommit(t *testing.T) {
	testWithTimeout(t, 10*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)
	provideReloadDB(root, "db", &reloadTestDB{gen: 1})
	MustInvokeNamed[*reloadTestDB](root, "db")

	releaseBuild := make(chan struct{})
	reloadDone := make(chan error, 1)

	go func() {
		reloadDone <- root.ReloadWithContext(context.Background(), func(i Injector) error {
			OverrideNamed(i, "db", func(_ Injector) (*reloadTestDB, error) {
				<-releaseBuild
				return &reloadTestDB{gen: 2}, nil
			})
			return nil
		})
	}()

	// 等待重载进入构造阶段（门禁已被写锁持有）
	time.Sleep(100 * time.Millisecond)

	invokeDone := make(chan *reloadTestDB, 1)
	go func() {
		db, _ := InvokeNamed[*reloadTestDB](root, "db")
		invokeDone <- db
	}()

	select {
	case <-invokeDone:
		t.Fatal("invoke must wait while reload is in progress")
	case <-time.After(100 * time.Millisecond):
	}

	close(releaseBuild)

	is.NoError(<-reloadDone)
	select {
	case db := <-invokeDone:
		is.Equal(2, db.gen)
	case <-time.After(2 * time.Second):
		t.Fatal("invoke did not unblock after reload")
	}
}

// 重载后 Clone：克隆体拿到新一轮注册（但不复制实例），可独立构造。
func TestRootScope_ReloadWithContext_CloneUsesNewRegistrations(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)
	provideReloadDB(root, "db", &reloadTestDB{gen: 1})
	MustInvokeNamed[*reloadTestDB](root, "db")

	err := root.ReloadWithContext(context.Background(), func(i Injector) error {
		OverrideNamed(i, "db", func(_ Injector) (*reloadTestDB, error) {
			return &reloadTestDB{gen: 2}, nil
		})
		return nil
	})
	is.NoError(err)

	clone := root.Clone()
	defer clone.Shutdown()

	clonedDB := MustInvokeNamed[*reloadTestDB](clone, "db")
	is.Equal(2, clonedDB.gen)
}

// 重载后 Shutdown：新实例被正常关闭；旧实例此前已关闭，不重复操作。
func TestRootScope_ReloadWithContext_ShutdownAfterReload(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	root := New()
	db1 := &reloadTestDB{gen: 1}
	provideReloadDB(root, "db", db1)
	MustInvokeNamed[*reloadTestDB](root, "db")

	db2 := &reloadTestDB{gen: 2}
	err := root.ReloadWithContext(context.Background(), func(i Injector) error {
		OverrideNamed(i, "db", func(_ Injector) (*reloadTestDB, error) {
			return db2, nil
		})
		return nil
	})
	is.NoError(err)

	report := root.Shutdown()
	is.True(report.Succeed, report.Error())
	is.True(db1.isClosed())
	is.True(db2.isClosed())
}

// 连续多轮重载：每轮都正确替换、关旧，失败轮不影响下一轮。
func TestRootScope_ReloadWithContext_MultipleRounds(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)
	provideReloadDB(root, "db", &reloadTestDB{gen: 1})
	MustInvokeNamed[*reloadTestDB](root, "db")

	override := func(gen int, healthErr error) error {
		return root.ReloadWithContext(context.Background(), func(i Injector) error {
			OverrideNamed(i, "db", func(_ Injector) (*reloadTestDB, error) {
				return &reloadTestDB{gen: gen, healthErr: healthErr}, nil
			})
			return nil
		})
	}

	is.Error(override(2, errors.New("round 2 unhealthy")))
	is.Equal(1, MustInvokeNamed[*reloadTestDB](root, "db").gen)

	is.NoError(override(3, nil))
	is.Equal(3, MustInvokeNamed[*reloadTestDB](root, "db").gen)

	is.Error(override(4, errors.New("round 4 unhealthy")))
	is.Equal(3, MustInvokeNamed[*reloadTestDB](root, "db").gen)

	is.NoError(override(5, nil))
	is.Equal(5, MustInvokeNamed[*reloadTestDB](root, "db").gen)
}

// 试构造时连带首次构造了未被替换的 lazy 依赖，回滚后该连带实例必须复位，
// 下一次 Invoke 重新构造，且调用观测恢复。
func TestRootScope_ReloadWithContext_RollbackResetsIncidentallyBuilt(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	root := newReloadTestRoot(t)

	cacheBuilds := int32(0)
	ProvideNamed(root, "cache", func(_ Injector) (*reloadTestDB, error) {
		atomic.AddInt32(&cacheBuilds, 1)
		return &reloadTestDB{gen: 100}, nil
	})
	ProvideNamed(root, "db", func(i Injector) (*reloadTestDB, error) {
		MustInvokeNamed[*reloadTestDB](i, "cache")
		return &reloadTestDB{gen: 1}, nil
	})

	// db、cache 均未调用
	err := root.ReloadWithContext(context.Background(), func(i Injector) error {
		OverrideNamed(i, "db", func(i Injector) (*reloadTestDB, error) {
			cache := MustInvokeNamed[*reloadTestDB](i, "cache")
			is.Equal(100, cache.gen)
			return nil, errors.New("db build failed")
		})
		return nil
	})
	is.Error(err)

	// cache 在失败的试构造中被连带首次构造；回滚后它必须复位为未构造。
	invoked := root.ListInvokedServices()
	for _, desc := range invoked {
		is.NotEqual("cache", desc.Service)
	}

	// 重新调用 cache 会再次构造（说明旧的连带实例已被关闭/复位）
	MustInvokeNamed[*reloadTestDB](root, "cache")
	is.Equal(int32(2), atomic.LoadInt32(&cacheBuilds))
}
