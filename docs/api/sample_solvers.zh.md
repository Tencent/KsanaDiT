# 采样求解器

扩散采样过程的 ODE/SDE 求解器。

::: kdit.sample_solvers
    options:
      members:
        - get_sample_scheduler
        - FlowMatchEulerScheduler
        - EulerScheduler
        - FlowDPMSolverMultistepScheduler
        - FlowUniPCMultistepScheduler
