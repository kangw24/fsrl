# CGR-v3.2 项目 Beamer

主文件：`cgr_v3_2_project_beamer.tex`

这份 16:9 中文 Beamer 共 9 页，按“研究方法与已有结果”的汇报格式组织，内容覆盖：

- Liu、Wang 与 Luo（2026）的行为任务与本仓库的 episode/input/RNG 契约；
- 当前 CGR-v3.2 的 anchor、随机关系编码、RLS 和共享全序读出；
- 人类与模型的终局指标、距离效应、个体化、自洽错误、双峰和模块消融；
- 实验流程、行为结果、功能模块及当前认知边界的简要对齐结论。

所有图表均由 TikZ/PGFPlots 在 TeX 内生成，不依赖外部图片。数值来自当前仓库的冻结开发报告。

## 编译

优先使用 XeLaTeX；脚本在 XeLaTeX 不可用时也支持 Tectonic。需要 `ctex`、`beamer`、`pgfplots`、`tikz` 和 `booktabs`。项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_cgr_beamer.ps1
```

脚本执行两遍 XeLaTeX，或调用 Tectonic 并自动完成一次重跑；最终 PDF 写入 `output/pdf/cgr_v3_2_project_beamer.pdf`。在 `pdftoppm` 可用时，逐页 PNG 写入 `tmp/pdfs/cgr_v3_2_project_beamer/rendered/`。Tectonic 分支会优先把 Windows 自带的 Noto CJK 字体复制到临时构建目录，不把字体文件加入版本控制。

当前参数和模型结果具有回溯开发性质；演示已明确区分直接事实、实现验证、前向预测与认知解释，不能将终局拟合表述为算法或神经机制同一性的证明。
