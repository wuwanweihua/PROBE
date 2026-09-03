# PROBE 12周详细实现步骤清单(v2,以 Pilot 为入口)

> 本文件为**自包含执行清单**:第1-2周 = 两周生死 Pilot(唯一入口),通过三关卡后才进入第3-12周(月2-3 完整系统 + 成稿)。若 pilot 不通过,第3周起按诊断分支执行(见第3周 Plan B),不进入主线。

---

## Pilot 三关卡判决口径(内联,第1-2周执行与第3周分流都依据此处)

- **关卡1(决定性)——同状态候选配对排序准确率 ≥65%,且 C 组须达到预注册最小增量 δ_CB=10pp**:C 组(K-sampling 行为分布)相对 B 组(single-forward hidden state)的配对排序准确率增量 ≥10pp 且 clustered bootstrap 95% CI 不跨 0。配对以 Beta-Binomial 后验确认(P(p_i>p_j)>0.9)而非单次标签定序;功效以**独立 base state** 为单位(≥85 个),不用同 state 相关配对充数。
- **关卡2(辅证)——校准/排序增量显著**:C 相对 max(A,B) 的 Brier/NLL 差或 AUROC 差显著(CI 不跨 0),不收点估计。
- **关卡3(方向性)——物理经济性方向一致 + 标定成本可回本**:选 probe 排名第一候选执行 vs 随机/默认候选,报告成功率差、物理失败次数差、CI;必须计"总交互成本 = 离线标定成本 + 在线执行成本",给出 break-even 方向。不做 p<0.05 决定。
- **三个必报指标(防"只评容易区分"的候选)**:条件化(仅有效配对)准确率之外,必须同时报 coverage–accuracy 曲线(以 Beta-Binomial 定序置信度扫描覆盖比例 vs 准确率)与全体候选 expected regret(预测第一名的候选与真实最优的 p̂ 差,含"不可分辨对",不做剔除)。
- **A/B/C 三组对照的评估口径**:同划分(GroupKFold 按 base_state 分组)、同目标(经验成功率 p̂)、同评估,显著性用 clustered bootstrap(以 base_state 为聚类单元,普通 McNemar/DeLong 因同 state 多候选不独立而失效)。
- **术语纪律**:在线选择机制称 "proxy-feedback adaptive selection";bootstrap/ensemble 是经验不确定性估计而非严格贝叶斯后验,未经验证前不宣称 standard BAI 理论保证。
- **生死线**:三关卡全过且 **C 显著超过 B** 才进入第3周主线。若只超过 observation-only(A)而不超过 single-forward(B),应放弃"行为分布是新增测量通道"这一核心主张,走诊断分支。

---

## 第1周(Pilot D1-D5):环境打通、采样器与采集器

**本周目标**:把 LIBERO-Long + π0.5 跑通到"可重复 rollout + 可批量采集",并出 wall-clock 预算表与 C>B 增量模拟功效分析——本周结束时必须知道"两周能不能采够 ≥85 个独立 base state"。

**具体步骤**:
1. 建 conda 环境,装 LIBERO(`git clone` + `pip install -e .`)与 π0.5 推理栈(openpi);下载 π0.5 LIBERO 微调 checkpoint(本机 HF 被墙,用 hf-mirror 或手动 scp)。中间产出:环境可 `import libero, openpi` 无报错。
2. 写单 episode 冒烟脚本 `scripts/smoke_rollout.py`:加载 LIBERO-Long 任一任务,π0.5 以标准温度跑一条完整 episode,打印观测/动作 shape 与成功标志;出视频+日志。中间产出:确认观测/动作接口正确。
3. 写 K 采样封装 `probe_vla/sampler.py`:给定 (observation, instruction),对 π0.5 前向 K=32 次(flow/diffusion 用不同噪声种子),返回 K×chunk_len×action_dim 张量;batch 化。**同时实测两个数字**:单次 rollout 秒数 `T_rollout`、单次 probe 前向秒数 `T_probe`。中间产出:采样函数 + 基准数字。
4. 种子流分离:在采集脚本里显式建 `probe_seed_stream` 与 `exec_seed_stream` 两条独立随机流,写入日志字段。中间产出:种子纪律落地,日志含两类种子。
5. **算 wall-clock 预算表**:总需求 ≈ 85-100 state × 3 条件 × N=8 ≈ 2000-2400 次 rollout + 约 8000 次 probe 前向;按 `(rollouts×T_rollout + probes×T_probe)/(并行环境数×GPU数)` 估算,判断 D5-D6 是否可行;若超可用时长 70%,按序降 N→6、每 state 条件→2、M→3。中间产出:`docs/timing_budget_v0.md`(含 T_rollout/T_probe、并行度、推算总时长、是否达标)。
6. **做 C>B 增量模拟功效分析**:按预期准确率与 state 内相关结构生成合成配对数据,跑 clustered bootstrap 数检出率,确认 85 state 是否足以在 power≥0.8 下检出 δ_CB=10pp;不足则给所需 state 数。中间产出:`docs/power_v0.md`(模拟设定、code、检出率曲线)。

**产出物**:`probe_vla/sampler.py`、`scripts/smoke_rollout.py`、`docs/timing_budget_v0.md`、`docs/power_v0.md`、环境可导入验证记录。

**完成判定标准**:smoke 跑通一条完整 episode、接口 shape 正确;`T_rollout` 与 `T_probe` 有数;预算表明确给出"两周采集规模该是多少";功效分析给出"85 state 能否检出 δ_CB=10pp"的明确结论(含需多少 state)。

**依赖关系**:无前置。若 D2 末 π0.5/LIBERO 仍无法批量 rollout(依赖冲突/checkpoint 不可得)→ 立即切 OpenVLA+LIBERO(不追 π0.5),第3周"第二底座"改为补 π0.5,其余不变。若 LIBERO seed/state 恢复接口不支持固定初态重置 → N 重复退化为"近似同状态"(report 中明确标注),这会抬高标签噪声:提高同初态重复次数的方差贡献,据此下调关卡2的期望(把"校准接近最优"的判据放宽或改为"与噪声上界可比"即算过),并同步下调第2周判断后的置信表述。

---

## 第2周(Pilot D6-D10):采集、三组预测器、三关卡

**本周目标**:采够独立 base state,训练 A/B/C 三组预测器,跑出关卡1(排序准确率 + δ_CB 差值检验)、关卡2(Brier/NLL 与 AUROC)、关卡3(经济性方向 + 标定成本),出一页决策备忘。

**具体步骤**:
1. 按第1周预算表定规模,全量采集:写 `scripts/collect_calls_v2.py`,固定初始状态可复现重置 + 条件枚举(措辞×技能,VLA-backed 候选 2-3 个,措辞语义等价经过模板/人工核对)+ N=8 重复;后台续跑至 ≥85 个独立 base state;先采 10 个 state 验证 p̂ 方差合理与种子纪律。中间产出:`data/calls_v0/`(每 state 一个 base_id 分组键,含 probe_seed/exec_seed 字段)。
2. 同时实现 A 组(observation-only:冻结 SigLIP/CLIP 图像特征 + 指令嵌入 → GBT)与 B 组(single-forward:冻结 VLA 前向一次取残差流表征 → 线性/浅层探针)。中间产出:`probe_vla/predictors.py` 含 A、B 两组。
3. 实现 C 组特征库 `probe_vla/features.py`(成对 DTW 统计、时间发散尖峰保留、首步可达域重叠、措辞敏感度、观测扰动脆性)与能力头 `probe_vla/competence_head.py`(GBT 回归 p̂,含 **ensemble** 以构造在线后验);GroupKFold 按 base_state 分组划分。中间产出:`data/features_v0.parquet`、能力头(含 ensemble)、噪声上界。
4. 关卡1:构造预注册配对集(每 state 1-2 对),Beta-Binomial 定序(P(p_i>p_j)>0.9 才定序),三组排序准确率 + **δ_CB 差值检验**(clustered bootstrap)+ **coverage–accuracy 曲线** + **全体候选 expected regret**(含不可分对)。中间产出:`results/gate1_ranking.csv`、coverage 曲线图、regret 表。
5. 关卡2:C 相对 max(A,B) 的 Brier/NLL 差与 AUROC 差(clustered bootstrap CI,不收点估计);报告标签噪声上界。中间产出:`results/gate2_calibration.csv`。
6. 关卡3:D 组 retry selector(2608.17484 式,同 GPU 预算)对照;adaptive-K vs K=32 对照;proxy-feedback 在线选择可行性预演(ensemble 区间覆盖率初检);**标定成本数字与回本方向**。中间产出:`results/gate3_economy.csv`、累计成本量级。
7. 汇总三关卡结论到一页 `docs/pilot_verdict.md`:继续/终止建议 + 依据 + 实测功率与预算执行情况。

**产出物**:`data/calls_v0/`、`probe_vla/predictors.py`、`probe_vla/features.py`、`probe_vla/competence_head.py`、`results/gate1_ranking.csv`、`results/gate2_calibration.csv`、`results/gate3_economy.csv`、`docs/pilot_verdict.md`、coverage 曲线图、expected regret 表。

**完成判定标准**:关卡1 排序准确率 ≥65% & **C−B ≥ δ_CB=10pp 且 CI 不跨 0** & C 在 expected regret 上同为最优;关卡2 Brier/NLL 或 AUROC 增量显著(CI 不跨 0);关卡3 方向与关卡1 一致且标定成本有回本方向。三关卡全过关卡才进入第3周主线。

**依赖关系**:依赖第1周采样器、预算表、功效分析、种子纪律。**关键风险(最高危)**:关卡1未过,或 C 未显著超过 B(尤其"C 只超 observation-only 不超 single-forward"——此时应放弃"行为分布是新增测量通道"主张)。**Plan B**:第3周起不进主线,改诊断分支——C 组特征与失败类型关联分析、自信地错刻画、A/B/C 信息重叠分解,随后提前进写作;砍掉月2 的规划器开发。

---

## 第3周(月2周1,前提:pilot通过):双模式接口与 proxy-feedback 规划器 v0

**本周目标**:定型 probe/execute 双模式 API,实现预算约束下的 proxy-feedback 规划器 v0,在 LIBERO-Long 上端到端跑通 ≥10 个 episode。

**具体步骤**:
1. 接口定型 `probe_vla/tool_interface.py`:`execute(instruction, budget) -> (trajectory, new_obs, success)` 与 `probe(instruction, obs, K) -> feature_vector`(obs 为当前真实观测,技能候选限定 VLA-backed);写 `docs/api.md`。中间产出:接口单元测试通过。
2. 规划器 v0 `probe_vla/planner.py`:预算约束 best-arm identification / proxy-feedback adaptive selection——每个候选(措辞×技能)是一个 arm,依 competence-head ensemble 的后验优势概率 P(p_i>p_j) 决定继续采样或停止执行当前最优候选;停止阈与预算由延迟约束标定。写提示词 `prompts/planner_v0.txt`(LLM 作编排器,含任务分解、工具列表、probe 语义说明)。中间产出:规划器 + 提示词。
3. 决策日志 `probe_vla/logger.py`:每次 probe/execute 记录输入、特征、预测概率分布、后验优势概率、LLM 决策理由,存结构化 JSON。
4. 端到端联调:选 LIBERO-Long 2 个长程任务各 5 个 episode,人工复盘日志确认决策链合理。中间产出:10 个 episode 完整决策日志。
5. adaptive-K 落地到在线(方差 K=8/16 收敛即停),记录与固定 K=32 的 GPU 前向次数差与成功率差。
6. 记录端到端耗时,与无 probe 版对比,建延迟基线。

**产出物**:`probe_vla/tool_interface.py`、`probe_vla/planner.py`、`probe_vla/logger.py`、`prompts/planner_v0.txt`、`docs/api.md`、10 个 episode 决策日志、延迟基线表。

**完成判定标准**:≥10 个 episode 无 crash 跑完;日志完整可追溯每次决策(含后验优势概率);≥3 例"probe 低分→换候选→执行成功"正向案例;episode 时长 ≤ 无 probe 基线 2 倍。

**依赖关系**:依赖第2周能力头(含 ensemble)与特征库;pilot 通过为前提。若第2周为"关卡1过、关卡2/3未过"分支 → 本周只做接口与规划器骨架,先修变现环节(候选生成质量、候选承诺策略)再用 pilot 数据小规模验证,不直接铺开 10 个 episode。若第3周 probe 延迟不可接受(单步>30s)→ adaptive-K 早停 + 措辞降 M=3 + 按后验优势概率贪心截断 probe 矩阵,占用第4周前 2 天。

---

## 第4周(月2周2):候选枚举、归因协议与能力场记忆

**本周目标**:实现措辞×技能二维 probe 矩阵的候选比较、反事实归因协议、能力场记忆(二维),并在注入式扰动实验验证归因准确率超 VLM 基线 ≥15 个百分点。

**具体步骤**:
1. 候选枚举 `probe_vla/candidates.py`:每子任务生成措辞改写 M=3-5(LLM 生成 + 语义等价核对 + 去重)、技能候选(可用 VLA-backed 调用/上下文配置列表),组合成二维 probe 矩阵,批量 probe 后按预测概率排序。中间产出:批量化实现(共享观测编码,控制延迟)。
2. 归因协议 `probe_vla/attribution.py`:失败后跑单维反事实——只换措辞、只换技能,哪一维使分布方差显著下降/预测概率显著回升即归因该维;两者都无差异则归"能力缺失";staging 维作为顺序排除项(物理重 staging 后复 probe)。判定用 bootstrap 置信区间。
3. 注入式扰动归因实验:三类已知根因各 ≥30 例——(a) 措辞扰动:替换为易混淆同义句;(b) 技能不匹配:分配 VLA 训练分布外的物体/技能;(c) 能力缺失:组合超出去分布。跑 probe 归因,同时跑 VLM 像素归因基线(GPT-4o/Qwen-VL 看失败帧)。中间产出:`results/attribution_confusion_matrix.csv`。
4. 能力场记忆 v0 `probe_vla/memory.py`:存 (特征向量→p̂) 对,FAISS 近邻检索;能力头在线增量更新;检索命中对校准增益小实验。中间产出:增益数字。
5. 累计成本触发:把离线标定成本 + 在线执行成本逐步累计,开始画随部署 episode 数的累计成本曲线雏形。

**产出物**:`probe_vla/candidates.py`、`probe_vla/attribution.py`、`probe_vla/memory.py`、`results/attribution_confusion_matrix.csv`、归因对比柱状图、累计成本曲线雏形。

**完成判定标准**:probe 归因三类平均准确率比 VLM 像素归因高 ≥15 个百分点;90 例全部有日志可复核;记忆检索增益有正向或持平的明确数字。

**依赖关系**:依赖第3周接口与规划器。若第3周触发 probe 延迟 Plan B 占用本期前 2 天,则压缩本周步骤3 到 20 例/类(记入限制),不砍归因协议本身。

---

## 第5周(月2周3):baseline 阵列与第一轮全量对比(系统级第二关)

**本周目标**:跑通全部 baseline,在 LIBERO-Long/Pro 上完成第一轮对比,验证 PROBE 相对去 probe 消融在成功率或物理步数上有统计显著优势。

**具体步骤**:
1. 去 probe 消融(最关键):同一规划器、同一提示词,禁用 probe 行动,只留执行反馈;≥3 seed × LIBERO-Long 全任务。
2. Harness VLA 复现(或官方代码):执行轨迹统计 operating range + 据此路由,与 PROBE 同执行预算。
3. OpenETA 式经验库、VoLo 式 VLM 监控 baseline:各实现核心机制简化忠实版,实现假设写入 `docs/baseline_notes.md`。
4. 世界模型评估器对照:τ0-WM 式——用现成 LIBERO 视频预测模型/或 VLM 打分替换 probe 通道,同 GPU 预算比较预测力与成本。中间产出:WM 评估器预测 AUROC。
5. 统一评测 harness `scripts/run_benchmark.py`:所有方法同一任务集、同一 seed 列表(≥3),统一记录成功率、物理步数、重试次数、碰撞次数、GPU 时间;出第一轮结果表 `results/main_table_v1.csv` 与成功率-物理成本 Pareto 图。

**产出物**:`baselines/`(5 个 baseline)、`scripts/run_benchmark.py`、`results/main_table_v1.csv`、Pareto 图 v1、`docs/baseline_notes.md`。

**完成判定标准**:PROBE vs 去 probe 消融,成功率或物理步数至少一项差异显著(≥3 seed,配对/聚类 bootstrap,p<0.05);所有方法在同一 harness 下产出,数字可由脚本+seed 复现。

**依赖关系**:依赖第3-4周完整系统。**关键风险(第二高,触发时间窗口=本周五)**:与去 probe 消融无显著差异(指标1好但系统级不涨)。**Plan B(第6周前2天)**:诊断规划器——用期望效用显式替换朴素阈值,核查决策翻转率(probe 是否真的改变决策);翻率高但收益平 → 能力头在规划分布下失准,用第4周记忆在线更新补校准;翻率低 → 修提示词/决策规则。若第6周中仍平,论文重心移到归因准确率(第4周)与探测有效性(pilot 结果),系统级如实报告为初步结果,第6周 RoboTwin 降级为只跑探测有效性跨环境验证。

---

## 第6周(月2周4):第二环境泛化与方法节写作

**本周目标**:在 RoboTwin 2.0 复现主结论方向一致性,启动 BEHAVIOR-1K 链式初态协议,完成论文方法节草稿。

**具体步骤**:
1. RoboTwin 2.0 环境搭建 + 适配层 `envs/robotwin_adapter.py`:复用 `tool_interface.py` 与新写观测/动作适配;选 5 个任务,采集 ≥200 条调用,验证能力头 few-shot 适配(每新任务 ≤20 条)后 AUROC≥0.7。
2. RoboTwin 全量对比:PROBE vs 去 probe 消融 vs 第5周最强 baseline,≥3 seed,复用 `run_benchmark.py`。中间产出:`results/robotwin_table.csv`。
3. BEHAVIOR-1K 链式初态协议启动:实现链式评测脚本(前一子任务终态为后一子任务初态),本周跑通协议 + 首批数据。中间产出:协议脚本。
4. 写方法节 `paper/sec3_method.tex`:四模块 + 信息流图(把 ASCII 图重画为矢量系统框图 `figures/system_overview.pdf`);自查无未定义符号、与实现一致。
5. 周五对照实现逐条核对方法节描述(特征公式、BAI/proxy-feedback 决策规则、记忆更新)与代码一致。

**产出物**:`envs/robotwin_adapter.py`、`results/robotwin_table.csv`、BEHAVIOR-1K 协议脚本、`paper/sec3_method.tex`、`figures/system_overview.pdf`。

**完成判定标准**:RoboTwin 上经济性优势方向与 LIBERO 一致(不要求同幅度);方法节自查清单(符号/公式/代码一致性)全过;链式协议跑通 ≥5 条链。

**依赖关系**:依赖第5周 harness 与主结果。若第5周触发系统级 Plan B → 本周 RoboTwin 只做跨环境探测有效性验证(采数据+AUROC),对比实验砍掉,省时间给第5周遗留诊断。

---

## 第7周(月3周1):消融矩阵与预算-收益曲线

**本周目标**:完成全部消融实验,拿到自适应采样预算-收益曲线(价值-信息权衡卖点直接证据),收口 BEHAVIOR-1K 链式实验。

**具体步骤**:
1. 特征组消融:去 DTW 组/去措辞敏感度/去脆性/只留单组,各测 AUROC 与端到端成功率,≥3 seed。中间产出:`results/ablation_features.csv`。
2. K 值扫描:K∈{4,8,16,32,64} 下 AUROC 与 probe 耗时;adaptive-K 对照(第3周已落地),画 AUROC-成本曲线,确定推荐 K。
3. 能力头容量扫描:LightGBM 深度/树数 + 2层 MLP 对照 + **ensemble 成员数**(决定在线后验稳定性),验证轻量头是否饱和。
4. BAI/代理反馈采样预算扫描:每子任务采样预算 ∈{0,1,3,5,10},测成功率与物理步数,画预算-收益曲线;预期边际递减,若非单调如实记录并分析。中间产出:`results/budget_curve.csv` + 图。
5. BEHAVIOR-1K 链式初态全量实验完成,并入主表。
6. 更新 `run_benchmark.py` 支持一键重跑任意消融。

**产出物**:`results/ablation_features.csv`、K 扫描图、容量扫描表(含 ensemble 数)、`results/budget_curve.csv` 与曲线图、BEHAVIOR-1K 结果、更新版 benchmark 脚本。

**完成判定标准**:每项消融 ≥3 seed;预算-收益曲线形态明确(边际递减或如实报告非单调+分析);全部消融可由脚本+配置文件一键复现。

**依赖关系**:依赖第5-6周主结果与 harness。若第6周 RoboTwin 未收口,其对比实验挤占本周步骤3(容量扫描降为附录级、单 seed)。

---

## 第8周(月3周2):图表定稿与实验节、相关工作节

**本周目标**:定稿全部主图主表,完成实验节与相关工作节草稿,数字逐一核对。

**具体步骤**:
1. 主图4张定稿:①探测有效性(pilot 排序准确率 + coverage-accuracy 曲线 + expected regret 合版);②经济性对比(成功率-物理成本 Pareto,多方法);③归因对比(混淆矩阵/柱状图);④预算-收益曲线。统一样式表 `figures/style.mplstyle`,由 `scripts/make_figures.py` 一键生成。
2. 主表3张:主结果表(LIBERO+RoboTwin+BEHAVIOR)、消融表、归因表;LaTeX 由脚本从 csv 自动生成(`scripts/make_tables.py`)。
3. 写实验节 `paper/sec4_experiments.tex`:4.1 探测有效性(含 pilot 结果)、4.2 主结果、4.3 消融、4.4 归因;每个数字与 results 逐一对账,建 `docs/number_audit.md`。
4. 写相关工作节 `paper/sec2_related.tex`:按 idea 文档 §4 边界组织(含 E-TTS/ParallelWorld 归入世界模型/TTS 类、2608.17484 专条、plan级 vs call级 细分、Reuse Before You Retrieve 的 headroom/retry selector 必比基线)。
5. 交叉检查:实验节每条 claim 标注支撑图表编号;把 **proxy-feedback** 与 BAI 区分写清(不宣称理论保证)。

**产出物**:4张主图 PDF、3张主表 tex、`paper/sec4_experiments.tex`、`paper/sec2_related.tex`、`scripts/make_figures.py`、`scripts/make_tables.py`、`docs/number_audit.md`。

**完成判定标准**:删除 figures/tables 后两脚本一键完整重生成;`number_audit.md` 每个论文数字有对应 csv 行号,核对无误;实验节 claim-图表映射无悬空。

**依赖关系**:依赖第7周全部结果。若第7周有个别消融未跑完,先用占位表格行文,数字后补,但三张主表数据必须齐。

---

## 第9周(月3周3):真机决策点与引言、结论

**本周目标**:周一做出真机做/不做的书面决策,完成引言、洞察阐述与结论初稿,论文达到可通读状态。

**具体步骤**:
1. **周一上午硬决策点**:检查仿真实验收口清单(主表、消融、归因、预算曲线、覆盖/few-shot 验证是否全部锁定 + 第3周库)是否锁定;任一未收口 → 书面放弃真机(`docs/realrobot_decision.md` 记录依据),本周全投写作;全锁定且预计 ≥2 周富余 → 启动真机。
2. (若做真机)真机流程:选 2 个桌面多阶段任务,部署 π0.5/OpenVLA 真机 checkpoint,每任务 10 trial PROBE vs 去 probe 消融,协议参考 ManipArena;周五前未完成一半 trial 即截断,已有数据进附录。
3. 写引言 `paper/sec1_intro.tex`:问题(失败是学习工具能力的唯一渠道)→ 洞察(VLA 是可反事实查询的测量仪器)→ 贡献三条;贡献与实验节数字一一对应。
4. 写结论与限制节初稿:如实列出触发过的 Plan B(两周 pilot 后的三关卡分流、few-shot 默认、staging 移出、C−B 未达 δ_CB 等)及其含义;把"行为分布是新增测量通道"的边界主张写成可被证伪的表述。
5. 全文串读:摘要到结论通读一遍,记录逻辑断点清单。

**产出物**:`docs/realrobot_decision.md`(含依据)、`paper/sec1_intro.tex`、结论与限制节草稿、(可选)真机结果、逻辑断点清单。

**完成判定标准**:真机决策有书面记录;论文全节齐备可从头通读无缺章;引言三条贡献均能指到实验节具体数字。

**依赖关系**:依赖第8周全部章节与图表。**风险窗口**:真机是唯一可整体砍掉项,不允许挤占写作——触发即周一收口检查,周中不留"再等等"余地。

---

## 第10周(月3周4):全文打磨与可复现仓库

**本周目标**:交付完整可投稿 PDF(8页+附录)与干净环境可跑通的代码仓库。

**具体步骤**:
1. claims-证据逐条对齐自查:摘要与引言每条主张建表,标注支撑实验编号与数字;删掉或弱化无支撑主张。中间产出:`docs/claims_audit.md`。
2. 限制与失败分析节定稿:整合全部 Plan B 触发记录、自信地错未召回部分、coverage-accuracy 曲线揭示的盲区、预算曲线非单调点、C−B 灰区结论。
3. 仓库整理:目录规范化(`probe_vla/` 核心库、`baselines/`、`scripts/`、`envs/`、`paper/`);写 README(安装、数据下载、quickstart:一条命令跑通单 episode+probe、一条命令重生成主表主图);requirements 锁版本;**用 pilot 的独立种子流约定复现采集**(probe_seed/exec_seed 写死)。
4. 干净环境验证:新建 conda 环境按 README 从零安装,跑通 quickstart;修坑回写 README。
5. 内部预审:请 1-2 位同事按审稿人视角过一遍,收集意见修订;摘要、图注、附录(实现细节、提示词全文、超参表、coverage 校验)定稿。
6. 编译最终 PDF,检查页数、引用完整性(idea 文档全部 arXiv 号入 bib)、图表引用无悬空。

**产出物**:最终论文 PDF(8页+附录)、`docs/claims_audit.md`、可复现仓库(README+quickstart 验证通过)、内部预审意见及修订记录。

**完成判定标准**:每条 claim 在 `claims_audit.md` 有实验数字支撑;干净环境 quickstart 一次跑通;PDF 编译零警告悬空引用;预审意见全处理或书面说明不处理原因。

**依赖关系**:依赖第8-9周全部章节。若第9周做了真机且延迟,真机内容降为附录一段+一表,不改主文结构;写作与仓库整理优先级高于任何补实验。

---

## 第11周(富余/扩展缓冲,按需分配)

**本周目标**:消化前 10 周可能遗留的扩展项——第二 VLA 底座、多环境泛化、ensemble 覆盖率校验、break-even 精确曲线;若主线已全收口,则用于扩展实验加固论文。

**具体步骤**(按遗留优先级排序,只做仍有时间窗口的):
1. 第二 VLA 底座(π0.5 若在第1周被 OpenVLA 顶替则此处补 π0.5,反之补 OpenVLA):复用 `collect_calls_v2.py` 采 ≥300 条,复算特征与 AUROC,并入 cross-底座 对照。中间产出:`data/calls_v2/`、第二底座 AUROC。
2. ensemble 区间经验覆盖率校验(若要在论文中把 proxy-feedback 升格):预测区间对 held-out p̂ 的覆盖频率,达标才可提"校准不确定性"。中间产出:`results/coverage_check.csv`。
3. break-even 精确曲线:按部署 episode 数画累计成本曲线,给出精确 break-even point。中间产出:`results/break_even.png`。
4. 若 p:真机已决定做,本周用于收口真机 trial 与数据整理。
5. 若前三周有消融降级为单 seed(容量扫描等),本周补足 ≥3 seed。

**产出物**:(按实际执行)第二底座数据与 AUROC、`results/coverage_check.csv`、`results/break_even.png`、补齐的多 seed 消融、真机数据(若做)。

**完成判定标准**:本周新增任何数字都可由脚本复现;未做项记入 `docs/deferred_items.md` 并说明为何不做(不影响主文 claim)。

**依赖关系**:依赖前 10 周全部收口。**限制**:本周不得引入新主张或新实验主线——只做加固/补齐,且任何扩展必须不与已定稿的主文 claim 冲突;若真机或某扩展挤占写作时间,宁可放弃该扩展。

---

## 第12周(终稿与投稿准备)

**本周目标**:交付投稿就绪的终稿 + 最终复现包,完成内部预审闭环。

**具体步骤**:
1. 基于第11周扩展结果,做最后一次 claims-证据对齐与数字更新(若有新增),重生成全部图表主表,确认 `number_audit.md` 无失配。
2. 终稿润色:摘要定稿、图注定稿、附录补全(实现细节、提示词、超参、coverage 校验、pilot 决策备忘引用)。
3. 投稿系统准备:按目标会议(CoRL/ICRA/ICLR)模板校订格式、页数、引用风格;写 cover letter 要点;补 reproducibility statement(数据集/checkpoint/hf-mirror 版权、计算资源、seed)。
4. 二轮内部预审:请同事按审稿人视角复评,收集意见,修订或书面说明不处理原因。
5. 打包:一键复现脚本 + README + 数据下载说明(若有上传)归档到可复现仓库;确认许可证与数据版权。

**产出物**:投稿终稿 PDF、reproducibility statement、二轮预审意见与处理记录、最终仓库归档(含 tag)。

**完成判定标准**:终稿页数/格式符合目标会议;每条 claim 有数字支撑且 `number_audit` 通过;代码仓库干净环境 quickstart 跑通;预审意见全部处理或书面说明。

**依赖关系**:依赖第10-11周成品。若第11周遗留扩展未完成,第12周不得再补实验,只做已有结果的打磨——**写作为最高优先级,任何未完成项记入限制/未来工作,不阻塞投稿**。

---

## 跨周风险总览(触发窗口精确到周)

| 风险 | 时间窗口 | 触发条件 | Plan B 及其后调整 |
|---|---|---|---|
| 环境/底座复现失败 | 第1周 D2 | D2 末无法批量 rollout | 切 OpenVLA+LIBERO;第11周改用 OpenVLA 为第二底座(补 π0.5 可选) |
| 固定初态不可复现 | 第1周 D3-D4 | LIBERO seed/state 恢复接口不支持 | N 重复退化为近似同状态,标注并下调关卡2期望;后续所有 state 级指标按近似处理 |
| 两周采集预算不足 | 第1周 D5 | 预算表显示超可用时长 70% | 按序降 N→6、每 state 条件→2、M→3;若仍不足,第2周减关卡1目标(报告功效-预算权衡) |
| **核心假设证伪(最高危)** | 第2周 D8-D10 | 关卡1未过,或 C−B < δ_CB=10pp(尤其 C 不超 B) | 第3周起走诊断分支,不进主线;砍月2 规划器,提前写作 |
| probe 延迟 | 第3周 | 单步决策>30s | adaptive-K 早停+措辞降 M=3+后验贪心截断;占用第4周前2天 |
| **有信号但系统不涨(次高危)** | 第5周五 / 第6周中 | 与去 probe 消融无显著差异 | 第6周前2天诊断规划器(期望效用+决策翻转率);若仍平,第6周 RoboTwin 降级探测有效性,论文重心移归因+探测有效性 |
| 与 B 组无增量(老生死线复检) | 第6周 | 系统级提升主要靠单次表征而非行为分布 | 明确"行为分布是新增通道"主张降级为实证观察,重写核心贡献 |
| 能力头校准在规划分布下失准 | 第6周 | 决策翻转率高但收益平 | 用第4周记忆在线更新补校准;仍无效则如实报告为限制 |
| 真机挤占写作 | 第9周周一 | 仿真未收口 | 放弃真机;论文限定仿真+讨论 |
| 第11周扩展挤占 | 第11周 | 任何扩展威胁写作 | 放弃该扩展,记入 deferred_items.md |
