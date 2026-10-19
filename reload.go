package do

import (
	"context"
	"fmt"
	"sync"
)

// ReloadWithContext 在不停止容器的前提下，原子地成批替换根作用域中已注册的服务，
// 适用于线上调整数据库连接、消息队列客户端以及依赖它们的处理器等长期运行服务。
//
// configure 收到一个临时注入器，可以、且应当只使用既有的 Override 系列函数
// （Override / OverrideNamed / OverrideValue / OverrideNamedValue /
// OverrideTransient / OverrideNamedTransient）一次性声明本轮要替换的多个服务。
// Override 只会记录到暂存区，在重载成功提交前不会影响任何线上注册。
// configure 的职责仅是声明注册变更，不应在其中承载业务逻辑；如确需读取当前
// 实例（例如基于旧值计算新配置），可以调用 Invoke，其结果不会影响原子性保证。
//
// 重载成功（configure 成功、新实例全部构造成功、健康检查全部通过）后：
//   - 本轮替换的服务包装器被整批安装，根作用域与既有子作用域中后续的 Invoke
//     都会解析到同一轮新配置，一次依赖解析不可能读到新旧两轮混用的依赖；
//   - 未被本轮替换的服务实例保持原样，不会被重建或关闭；
//   - 受影响的旧实例按 DAG 依赖顺序关闭（依赖方先于被依赖方）。
//
// 以下任一情况都会中止重载并完整回滚，原注册与旧实例继续可用，且允许随后重试：
//   - configure 返回错误（或 panic）；
//   - 新服务构造失败；
//   - 新服务健康检查失败；
//   - ctx 在提交前被取消。
//
// 回滚会关闭本轮已经创建出来的新实例（含 transient 的试构造实例）。
// 重载期间并发的 Invoke 会在门禁处等待，直到本轮重载成功提交或回滚完成，
// 因此读取方既不会看到新旧依赖混用，也不会因为新配置构造失败而失去现有服务。
//
// 注意：处理器等单例若通过构造函数注入了被替换的服务，需要在同一轮 configure
// 中一并 Override，否则已构造的单例仍持有旧依赖，这符合“未受影响实例保持原样”
// 的语义。
func (s *RootScope) ReloadWithContext(ctx context.Context, configure func(Injector) error) error {
	if configure == nil {
		return fmt.Errorf("DI: reload configure function cannot be nil")
	}
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("DI: reload aborted before start: %w", err)
	}

	// 与 Shutdown / Clone 互斥，保证没有其他整体性操作与本轮重载交错。
	s.lifecycleMu.Lock()
	defer s.lifecycleMu.Unlock()

	// 冻结依赖解析：等待所有在途 Invoke 结束后独占配置视图。
	// configure 暂存、试构造、健康检查与提交期间，外部读取方都会在门前等待，
	// 因此它们要么只看到旧轮、要么只看到新轮，绝不会看到中间态。
	s.resolutionMu.Lock()
	defer s.resolutionMu.Unlock()

	session := newReloadSession()
	staging := newReloadStagingInjector(s, session)

	if err := runReloadConfigure(configure, staging); err != nil {
		// 暂存内容尚未安装到正式映射，不存在需要清理的容器实例。
		return err
	}
	if len(session.order) == 0 {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("DI: reload aborted before install: %w", err)
	}

	// 安装前快照：所有作用域的已调用集合（用于回滚观测恢复），
	// 以及每个被替换服务的旧包装器。
	invokedBefore := snapshotAllInvokedServiceNames(s.self)
	pendings := make([]*reloadPending, 0, len(session.order))
	for _, rep := range session.order {
		oldWrapper, existed := rep.scope.serviceGet(rep.name)
		pendings = append(pendings, &reloadPending{
			rep:        rep,
			oldWrapper: oldWrapper,
			existed:    existed,
		})
	}

	// 整批安装：同一轮新配置一次性进入各作用域的正式服务映射。
	for _, p := range pendings {
		p.rep.scope.serviceSet(p.rep.name, p.rep.wrapper)
	}

	// 试构造全部新服务并执行健康检查。失败即回滚。
	built, err := s.validateReloadServices(ctx, session)
	if err != nil {
		s.rollbackReload(pendings, invokedBefore, built)
		return err
	}

	// 提交成功：关闭受影响的旧实例；transient 的试构造实例属于一次性验证产物，
	// 同样需要关闭。关闭顺序按 DAG 依赖关系排列（依赖方先关）。
	s.shutdownOldInstancesAfterCommit(pendings)
	s.shutdownTransientTrials(built)

	return nil
}

// runReloadConfigure 执行用户提供的 configure，并把 panic 转换为错误，
// 保证 configure 中的异常也不会破坏现有注册。
func runReloadConfigure(configure func(Injector) error, staging Injector) (err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(error); ok {
				err = e
			} else {
				err = fmt.Errorf("DI: reload configure panicked: %v", r)
			}
		}
	}()

	return configure(staging)
}

// reloadReplacement 记录 configure 中对某个作用域下某个服务名的一次替换。
type reloadReplacement struct {
	desc    ServiceDescription // 由作用域与服务名构成的全局唯一定位
	scope   *Scope             // 替换生效的作用域（重载入口固定为根作用域）
	name    string             // 被替换的服务名
	wrapper any                // 新的服务包装器，提交前不进入正式映射
}

// reloadPending 表示一个已进入安装阶段的替换，携带旧包装器以便回滚。
type reloadPending struct {
	rep        *reloadReplacement
	oldWrapper any
	existed    bool
}

// reloadSession 是单次重载的暂存上下文。
type reloadSession struct {
	order []*reloadReplacement                      // 保序，保证构造与健康检查顺序稳定
	byKey map[ServiceDescription]*reloadReplacement // 同名重复 Override 时以后写为准
}

func newReloadSession() *reloadSession {
	return &reloadSession{
		order: []*reloadReplacement{},
		byKey: map[ServiceDescription]*reloadReplacement{},
	}
}

// record 录制一次 serviceSet（Override 系列最终都会走到这里）。
func (s *reloadSession) record(scope *Scope, name string, wrapper any) {
	desc := newServiceDescription(scope.id, scope.name, name)
	if rep, ok := s.byKey[desc]; ok {
		// configure 内多次 Override 同一个服务：最后一次生效。
		rep.wrapper = wrapper
		return
	}

	rep := &reloadReplacement{
		desc:    desc,
		scope:   scope,
		name:    name,
		wrapper: wrapper,
	}
	s.byKey[desc] = rep
	s.order = append(s.order, rep)
}

// reloadStagingInjector 是传给 configure 的临时注入器。
// 写操作（serviceSet）只落入暂存会话；读操作返回“暂存覆盖 + 线上旧注册”的
// 合并视图，因此 configure 中的重复注册校验、AsNamed 存在性判断等行为与
// 直接操作根作用域保持一致，而线上映射在提交前完全不受影响。
type reloadStagingInjector struct {
	Injector // 委托根作用域的全部其他能力

	session *reloadSession
	scope   *Scope
}

func newReloadStagingInjector(root *RootScope, session *reloadSession) *reloadStagingInjector {
	return &reloadStagingInjector{
		Injector: root,
		session:  session,
		scope:    root.self,
	}
}

func (i *reloadStagingInjector) serviceSet(name string, service any) {
	i.session.record(i.scope, name, service)
}

func (i *reloadStagingInjector) serviceExist(name string) bool {
	if _, ok := i.session.byKey[newServiceDescription(i.scope.id, i.scope.name, name)]; ok {
		return true
	}

	return i.scope.serviceExist(name)
}

func (i *reloadStagingInjector) serviceGet(name string) (any, bool) {
	if rep, ok := i.session.byKey[newServiceDescription(i.scope.id, i.scope.name, name)]; ok {
		return rep.wrapper, true
	}

	return i.scope.serviceGet(name)
}

func (i *reloadStagingInjector) serviceGetRec(name string) (any, *Scope, bool) {
	if rep, ok := i.session.byKey[newServiceDescription(i.scope.id, i.scope.name, name)]; ok {
		return rep.wrapper, i.scope, true
	}

	return i.scope.serviceGetRec(name)
}

func (i *reloadStagingInjector) serviceForEach(cb func(string, *Scope, any) bool) {
	seen := map[string]struct{}{}

	for _, rep := range i.session.order {
		if rep.scope != i.scope {
			continue
		}

		seen[rep.name] = struct{}{}
		if !cb(rep.name, i.scope, rep.wrapper) {
			return
		}
	}

	i.scope.serviceForEach(func(name string, scope *Scope, svc any) bool {
		if _, ok := seen[name]; ok {
			return true // 已被暂存版本覆盖
		}

		return cb(name, scope, svc)
	})
}

// serviceForEachRec 与根作用域正式行为一致：根作用域没有祖先，遍历自身即可。
func (i *reloadStagingInjector) serviceForEachRec(cb func(string, *Scope, any) bool) {
	i.serviceForEach(cb)
}

// reloadBuiltInstance 记录试构造阶段实际创建出来、回滚时需要清理的实例。
type reloadBuiltInstance struct {
	scope   *Scope
	name    string
	wrapper serviceWrapperAny
	trial   any // 仅 transient：试构造产生的一次性实例
}

// validateReloadServices 整批安装后，在持有重载门禁的情况下试构造所有被替换的
// 服务包装器，并对实现了 Healthchecker 的实例做健康检查。
// 解析链路通过 guarded 标记复用当前写锁，因此构造函数中的嵌套 Invoke 不会
// 重复加锁，并且只会读取本轮新配置。
//
// 返回错误时，built 可能包含部分已构造实例，调用方必须执行回滚清理。
func (s *RootScope) validateReloadServices(ctx context.Context, session *reloadSession) ([]*reloadBuiltInstance, error) {
	if s.opts.HealthCheckGlobalTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, s.opts.HealthCheckGlobalTimeout)
		defer cancel()
	}

	built := make([]*reloadBuiltInstance, 0, len(session.order))

	for _, rep := range session.order {
		if err := ctx.Err(); err != nil {
			return built, fmt.Errorf("DI: reload aborted while building services: %w", err)
		}

		wrapper, ok := rep.wrapper.(serviceWrapperAny)
		if !ok {
			return built, fmt.Errorf("DI: service `%s` is not a valid service wrapper", rep.name)
		}

		item := &reloadBuiltInstance{
			scope:   rep.scope,
			name:    rep.name,
			wrapper: wrapper,
		}

		if wrapper.getServiceType() == ServiceTypeTransient {
			// transient 不缓存实例：试构造一次以验证工厂可用，实例随后即弃。
			trial, err := invokeAnyByNameGuarded(rep.scope, rep.name, true)
			if err != nil {
				return built, err
			}
			item.trial = trial
		} else {
			// lazy / eager / alias 都经由标准解析路径构造，
			// 以保证钩子、DAG 边、调用记录与真实 Invoke 完全一致。
			if _, err := invokeAnyByNameGuarded(rep.scope, rep.name, true); err != nil {
				return built, err
			}
		}

		built = append(built, item)
	}

	if err := s.reloadHealthCheck(ctx, built); err != nil {
		return built, err
	}

	return built, nil
}

// reloadHealthCheck 并行健康检查本轮新建实例。
// 单个服务的超时遵循 InjectorOpts.HealthCheckTimeout，
// 整体超时遵循 InjectorOpts.HealthCheckGlobalTimeout（在调用方包装）。
func (s *RootScope) reloadHealthCheck(ctx context.Context, built []*reloadBuiltInstance) error {
	type healthcheckCandidate struct {
		service string
		wrapper serviceWrapperHealthcheck
		trial   any // 仅 transient 试构造实例
	}

	candidates := make([]healthcheckCandidate, 0, len(built))
	for _, item := range built {
		if item.wrapper.getServiceType() == ServiceTypeTransient {
			if instanceHealthchecker(item.trial) {
				candidates = append(candidates, healthcheckCandidate{service: item.name, trial: item.trial})
			}
			continue
		}

		if item.wrapper.isHealthchecker() {
			candidates = append(candidates, healthcheckCandidate{
				service: item.name,
				wrapper: item.wrapper.(serviceWrapperHealthcheck), //nolint:errcheck,forcetypeassert
			})
		}
	}

	if len(candidates) == 0 {
		return nil
	}

	checkCtx, cancelCheck := context.WithCancel(ctx)
	defer cancelCheck()

	var (
		mu       sync.Mutex
		firstErr error
	)

	var wg sync.WaitGroup
	for _, candidate := range candidates {
		wg.Add(1)

		go func(c healthcheckCandidate) {
			defer wg.Done()

			serviceCtx := checkCtx
			if s.opts.HealthCheckTimeout > 0 {
				var cancel context.CancelFunc
				serviceCtx, cancel = context.WithTimeout(checkCtx, s.opts.HealthCheckTimeout)
				defer cancel()
			}

			err := raceWithTimeout(serviceCtx, func(ctx context.Context) error {
				if c.wrapper != nil {
					return c.wrapper.healthcheck(ctx)
				}
				return healthcheckReloadInstance(c.trial, ctx)
			})
			if err == nil {
				return
			}

			mu.Lock()
			if firstErr == nil {
				firstErr = fmt.Errorf("DI: reload aborted, health check failed for service `%s`: %w", c.service, err)
				cancelCheck() // 尽快终止其余检查
			}
			mu.Unlock()
		}(candidate)
	}
	wg.Wait()

	// 上下文取消优先于单个健康检查错误：调用方据此识别“可重试的取消”。
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("DI: reload aborted during health checks: %w", err)
	}

	if firstErr != nil {
		return firstErr
	}

	return nil
}

// rollbackReload 在试构造或健康检查失败（含上下文取消）后完整恢复旧状态：
//  1. 恢复旧包装器（或删除本轮新增的注册），映射立即只呈现旧轮配置；
//  2. 关闭本轮新创建的包装器实例与 transient 试构造实例；
//  3. 恢复调用观测：试构造过程中被连带首次构造的未替换 lazy 服务要复位，
//     新增的调用记录要移除。
//
// 清理使用 context.Background()：即便用户 ctx 已取消，也必须尽力释放资源，
// 避免失败重载留下泄漏的新实例。
func (s *RootScope) rollbackReload(
	pendings []*reloadPending,
	invokedBefore map[string]map[string]struct{},
	built []*reloadBuiltInstance,
) {
	// 1. 恢复注册映射。
	for _, p := range pendings {
		if p.existed {
			p.rep.scope.serviceSet(p.rep.name, p.oldWrapper)
		} else {
			p.rep.scope.serviceUnset(p.rep.name)
		}
	}

	// 2. 关闭本轮新建实例（此时它们在正式映射中已不可达）。
	candidates := make([]*shutdownCandidate, 0, len(built))
	for _, item := range built {
		if item.wrapper.getServiceType() == ServiceTypeTransient {
			candidates = append(candidates, &shutdownCandidate{
				desc:  newServiceDescription(item.scope.id, item.scope.name, item.name),
				scope: item.scope,
				trial: item.trial,
			})
			continue
		}

		if !item.wrapper.isShutdowner() {
			continue
		}

		candidates = append(candidates, &shutdownCandidate{
			desc:    newServiceDescription(item.scope.id, item.scope.name, item.name),
			scope:   item.scope,
			wrapper: item.wrapper,
		})
	}
	s.shutdownCandidates(context.Background(), candidates)

	// 3. 恢复调用观测与连带构造的未替换 lazy 实例。
	s.resetIncidentallyBuiltServices(invokedBefore)
}

// resetIncidentallyBuiltServices 比对试构造前后的“已调用服务”集合：
// 试构造替换服务（如 alias）时，可能连带首次构造了未被替换的 lazy 目标。
// 回滚后这些实例也必须复位为未构造状态，才算真正恢复重载前的容器。
func (s *RootScope) resetIncidentallyBuiltServices(invokedBefore map[string]map[string]struct{}) {
	scopes := s.collectScopesByID()

	for scopeID, before := range invokedBefore {
		scope, ok := scopes[scopeID]
		if !ok {
			continue
		}

		after := scope.invokedServiceNamesSnapshot()
		newlyInvoked := make([]string, 0)
		for name := range after {
			if _, existedBefore := before[name]; !existedBefore {
				newlyInvoked = append(newlyInvoked, name)
			}
		}

		if len(newlyInvoked) == 0 {
			continue
		}

		candidates := make([]*shutdownCandidate, 0, len(newlyInvoked))
		for _, name := range newlyInvoked {
			serviceAny, ok := scope.serviceGet(name)
			if !ok {
				continue
			}

			// 只有 lazy 存在“此前未构造、试构造期间才构造”的概念；
			// eager 实例在注册时即存在，alias 的关停会波及共享目标，均跳过。
			wrapper, ok := serviceAny.(serviceWrapperAny)
			if !ok || wrapper.getServiceType() != ServiceTypeLazy {
				continue
			}

			buildTime, ok := serviceAny.(serviceWrapperBuildTime)
			if !ok {
				continue
			}
			if _, built := buildTime.getBuildTime(); !built {
				continue
			}
			if !wrapper.isShutdowner() {
				continue
			}

			candidates = append(candidates, &shutdownCandidate{
				desc:    newServiceDescription(scope.id, scope.name, name),
				scope:   scope,
				wrapper: wrapper,
			})
		}

		// 先复位连带构造的实例，再抹除调用记录。
		s.shutdownCandidates(context.Background(), candidates)
		scope.removeInvokedServiceNames(stringSliceToSet(newlyInvoked))
	}
}

// shutdownOldInstancesAfterCommit 在提交成功后关闭受影响的旧实例：
//   - 此前未构造的 lazy：没有实例需要关闭；
//   - alias：关停会转发到可能仍存活的共享目标，跳过；
//   - transient：无常驻实例，跳过；
//   - eager 与已构造的 lazy：仅当实现 Shutdowner 时才关闭。
func (s *RootScope) shutdownOldInstancesAfterCommit(pendings []*reloadPending) {
	candidates := make([]*shutdownCandidate, 0, len(pendings))

	for _, p := range pendings {
		wrapper, ok := p.oldWrapper.(serviceWrapperAny)
		if !ok {
			continue
		}

		switch wrapper.getServiceType() {
		case ServiceTypeAlias, ServiceTypeTransient:
			continue
		case ServiceTypeLazy:
			buildTime, ok := p.oldWrapper.(serviceWrapperBuildTime)
			if !ok {
				continue
			}
			if _, built := buildTime.getBuildTime(); !built {
				continue
			}
		}

		if !wrapper.isShutdowner() {
			continue
		}

		candidates = append(candidates, &shutdownCandidate{
			desc:    p.rep.desc,
			scope:   p.rep.scope,
			wrapper: wrapper,
		})
	}

	s.shutdownCandidates(context.Background(), candidates)
}

// shutdownTransientTrials 关闭健康检查通过后仍需释放的 transient 试构造实例。
func (s *RootScope) shutdownTransientTrials(built []*reloadBuiltInstance) {
	candidates := make([]*shutdownCandidate, 0, len(built))

	for _, item := range built {
		if item.wrapper.getServiceType() != ServiceTypeTransient {
			continue
		}

		candidates = append(candidates, &shutdownCandidate{
			desc:  newServiceDescription(item.scope.id, item.scope.name, item.name),
			scope: item.scope,
			trial: item.trial,
		})
	}

	s.shutdownCandidates(context.Background(), candidates)
}

// shutdownCandidate 是待关闭实例的统一描述：wrapper 指向已脱离正式映射的包装器，
// trial 指向 transient 的一次性实例。
type shutdownCandidate struct {
	desc    ServiceDescription
	scope   *Scope
	wrapper serviceWrapperAny
	trial   any
}

// shutdownCandidates 按 DAG 依赖顺序分批并行关闭候选实例：
// 每一轮选出“没有仍在等待的依赖方”的服务并行关闭，循环直至全部关闭。
// 若因为环依赖或仍有存活依赖方而无法推进，则退化为无序关闭（与容器整体
// Shutdown 的环依赖兜底策略保持一致）。
//
// 该方法直接持有包装器引用，不经过 serviceShutdown，因此不会改动正式映射与
// DAG 节点（重载前后作用域与服务名不变，DAG 边对新包装器依然成立）。
func (s *RootScope) shutdownCandidates(ctx context.Context, candidates []*shutdownCandidate) {
	scopeByID := s.collectScopesByID()

	remaining := make(map[ServiceDescription]*shutdownCandidate, len(candidates))
	for _, candidate := range candidates {
		remaining[candidate.desc] = candidate
	}

	for len(remaining) > 0 {
		s.dag.mu.RLock()
		pick := make([]*shutdownCandidate, 0, len(remaining))
		for desc, candidate := range remaining {
			blocked := false
			for dependent := range s.dag.dependents[desc] {
				if _, inSet := remaining[dependent]; inSet {
					// 依赖方同样等待关闭，它必须先关。
					blocked = true
					break
				}
				// 依赖方仍在容器中存活（本轮未替换）：优先等待，稍后由兜底处理。
				if owner := scopeByID[dependent.ScopeID]; owner != nil && owner.serviceExist(dependent.Service) {
					blocked = true
					break
				}
			}

			if !blocked {
				pick = append(pick, candidate)
			}
		}
		s.dag.mu.RUnlock()

		if len(pick) == 0 {
			// 环依赖或存活依赖方兜底：忽略顺序，关闭剩余全部。
			pick = pick[:0]
			for _, candidate := range remaining {
				pick = append(pick, candidate)
			}
		}

		var wg sync.WaitGroup
		for _, candidate := range pick {
			delete(remaining, candidate.desc)

			wg.Add(1)
			go func(c *shutdownCandidate) {
				defer wg.Done()
				s.shutdownOneCandidate(ctx, c)
			}(candidate)
		}
		wg.Wait()
	}
}

// shutdownOneCandidate 关闭单个候选实例，并触发关停钩子、记录关停错误日志。
func (s *RootScope) shutdownOneCandidate(ctx context.Context, candidate *shutdownCandidate) {
	// transient 试构造实例不归属任何包装器，直接按实例接口释放。
	if candidate.trial != nil {
		_ = shutdownReloadInstance(ctx, candidate.trial)
		return
	}

	shutdowner, ok := candidate.wrapper.(serviceWrapperShutdown)
	if !ok {
		return
	}

	s.opts.onBeforeShutdown(candidate.scope, candidate.desc.Service)
	err := shutdowner.shutdown(ctx)
	s.opts.onAfterShutdown(candidate.scope, candidate.desc.Service, err)
	if err != nil {
		s.opts.Logf("DI <scope=%s>: reload: failed to shutdown old instance of service %s: %v",
			candidate.scope.name, candidate.desc.Service, err)
	}
}

// collectScopesByID 收集根作用域及其全部后代作用域。
func (s *RootScope) collectScopesByID() map[string]*Scope {
	out := map[string]*Scope{}

	var walk func(*Scope)
	walk = func(scope *Scope) {
		out[scope.id] = scope
		for _, child := range scope.Children() {
			walk(child)
		}
	}
	walk(s.self)

	return out
}

// snapshotAllInvokedServiceNames 快照整棵作用域树的已调用服务名集合，
// 以作用域 ID 为键，供重载回滚时恢复调用观测。
func snapshotAllInvokedServiceNames(root *Scope) map[string]map[string]struct{} {
	out := map[string]map[string]struct{}{}

	var walk func(*Scope)
	walk = func(scope *Scope) {
		out[scope.id] = scope.invokedServiceNamesSnapshot()
		for _, child := range scope.Children() {
			walk(child)
		}
	}
	walk(root)

	return out
}

func stringSliceToSet(items []string) map[string]struct{} {
	out := make(map[string]struct{}, len(items))
	for _, item := range items {
		out[item] = struct{}{}
	}
	return out
}

// instanceHealthchecker 判断实例自身是否实现任一健康检查接口。
// 用于 transient 试构造实例（其包装器不承载健康检查能力）。
func instanceHealthchecker(instance any) bool {
	switch any(instance).(type) {
	case HealthcheckerWithContext, Healthchecker:
		return true
	default:
		return false
	}
}

// healthcheckReloadInstance 直接对实例执行健康检查，接口优先级与
// serviceEager/serviceLazy 保持一致。
func healthcheckReloadInstance(instance any, ctx context.Context) error {
	switch v := instance.(type) {
	case HealthcheckerWithContext:
		if err := ctx.Err(); err != nil {
			return err
		}
		return v.HealthCheck(ctx)
	case Healthchecker:
		if err := ctx.Err(); err != nil {
			return err
		}
		return v.HealthCheck()
	}
	return nil
}

// shutdownReloadInstance 直接按实例实现的关停接口释放资源，
// 用于 transient 试构造实例等脱离包装器管理的对象。
func shutdownReloadInstance(ctx context.Context, instance any) error {
	switch v := instance.(type) {
	case ShutdownerWithContextAndError:
		if err := ctx.Err(); err != nil {
			return err
		}
		return v.Shutdown(ctx)
	case ShutdownerWithError:
		if err := ctx.Err(); err != nil {
			return err
		}
		return v.Shutdown()
	case ShutdownerWithContext:
		if err := ctx.Err(); err != nil {
			return err
		}
		v.Shutdown(ctx)
	case Shutdowner:
		if err := ctx.Err(); err != nil {
			return err
		}
		v.Shutdown()
	}
	return nil
}
