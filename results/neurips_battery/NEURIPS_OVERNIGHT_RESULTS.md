# NeurIPS Overnight Battery Results

Generated: 2026-04-05 12:50:40
Total runtime: 10.7h
Total runs: 731


## exp1_multitask

- **collision_dinov2_continuous**: acc=95.6%±1.8% PD=0.509±0.069 CS=0.007±0.014
- **collision_dinov2_discrete**: acc=93.5%±1.6% PD=0.583±0.125 CS=0.061±0.090
- **collision_dinov2_raw_probe**: acc=95.0%±1.2% PD=0.986±0.013 CS=0.000±0.000
- **collision_vjepa2_continuous**: acc=0.0%±0.0% PD=0.000±0.000 CS=0.000±0.000
- **collision_vjepa2_discrete**: acc=0.0%±0.0% PD=0.000±0.000 CS=0.000±0.000
- **collision_vjepa2_raw_probe**: acc=0.0%±0.0% PD=0.000±0.000 CS=0.000±0.000
- **fall_dinov2_continuous**: acc=83.0%±7.2% PD=0.387±0.070 CS=0.000±0.000
- **fall_dinov2_discrete**: acc=77.3%±11.2% PD=0.474±0.181 CS=0.008±0.029
- **fall_dinov2_raw_probe**: acc=81.7%±6.7% PD=0.576±0.038 CS=0.000±0.000
- **fall_vjepa2_continuous**: acc=88.0%±7.4% PD=0.378±0.075 CS=0.003±0.008
- **fall_vjepa2_discrete**: acc=83.4%±9.0% PD=0.534±0.088 CS=0.024±0.063
- **fall_vjepa2_raw_probe**: acc=89.3%±6.0% PD=0.516±0.072 CS=0.003±0.007
- **ramp_dinov2_continuous**: acc=91.5%±4.2% PD=0.380±0.073 CS=0.000±0.000
- **ramp_dinov2_discrete**: acc=84.9%±13.4% PD=0.381±0.148 CS=0.032±0.077
- **ramp_dinov2_raw_probe**: acc=90.6%±5.0% PD=0.506±0.051 CS=0.000±0.000
- **ramp_vjepa2_continuous**: acc=80.8%±5.1% PD=0.311±0.073 CS=0.000±0.000
- **ramp_vjepa2_discrete**: acc=76.7%±8.7% PD=0.297±0.095 CS=0.079±0.114
- **ramp_vjepa2_raw_probe**: acc=80.7%±4.7% PD=0.453±0.087 CS=0.003±0.008
- **spring_clip_continuous**: acc=63.1%±14.9% PD=0.455±0.088 CS=0.026±0.054
- **spring_clip_discrete**: acc=63.8%±11.3% PD=0.499±0.236 CS=0.175±0.203
- **spring_clip_raw_probe**: acc=64.0%±13.9% PD=0.505±0.156 CS=0.034±0.055
- **spring_dinov2_continuous**: acc=73.7%±13.0% PD=0.523±0.094 CS=0.000±0.000
- **spring_dinov2_discrete**: acc=74.4%±12.6% PD=0.541±0.202 CS=0.096±0.119
- **spring_dinov2_raw_probe**: acc=73.9%±12.4% PD=0.661±0.121 CS=0.005±0.017
- **spring_vjepa2_continuous**: acc=83.5%±6.5% PD=0.529±0.082 CS=0.006±0.018
- **spring_vjepa2_discrete**: acc=82.5%±7.4% PD=0.717±0.045 CS=0.125±0.100
- **spring_vjepa2_raw_probe**: acc=83.2%±5.9% PD=0.700±0.037 CS=0.000±0.000

## exp2_vocab_sweep

- **K=16**: acc=75.0%±14.4% PD=0.540±0.244 CS=0.097±0.123
- **K=3**: acc=84.1%±6.0% PD=0.753±0.048 CS=0.170±0.096
- **K=32**: acc=68.4%±14.7% PD=0.453±0.295 CS=0.020±0.056
- **K=5**: acc=82.5%±7.4% PD=0.717±0.045 CS=0.125±0.100
- **K=64**: acc=60.2%±11.2% PD=0.388±0.273 CS=0.000±0.000
- **K=8**: acc=79.6%±12.3% PD=0.609±0.199 CS=0.094±0.102

## exp4_agent_sweep

- **N=1**: acc=84.2%±5.2% PD=0.762±0.047 CS=-0.112±0.916
- **N=16**: acc=73.4%±14.6% PD=0.511±0.165 CS=0.003±0.010
- **N=2**: acc=82.5%±5.8% PD=0.743±0.057 CS=0.291±0.383
- **N=4**: acc=82.0%±8.0% PD=0.715±0.050 CS=0.143±0.112
- **N=8**: acc=76.5%±14.0% PD=0.594±0.196 CS=0.034±0.069

## exp5_transfer

- **dinov2→dinov2**: acc=75.6%±12.9% (10 seeds)
- **vjepa2→vjepa2**: acc=82.0%±8.0% (10 seeds)

## exp6_beyond_physics

- **visual_continuous**: acc=68.5%±12.2% PD=0.536±0.136 CS=0.000±0.000
- **visual_discrete**: acc=68.9%±9.2% PD=0.608±0.221 CS=0.085±0.080
- **visual_raw_probe**: acc=68.2%±11.5% PD=0.665±0.145 CS=0.000±0.000

## exp7_faithfulness

- **continuous_native**: 84.3%±5.9%
- **continuous_transfer**: 58.9%±2.3%
- **discrete_native**: 82.0%±8.0%
- **discrete_transfer**: 68.7%±3.2%

## exp8_protocol_reuse

- **spring→collision**: acc=0.0%±0.0% (10 seeds)
- **spring→fall**: acc=65.8%±2.1% (10 seeds)
- **spring→ramp**: acc=59.2%±3.4% (10 seeds)
- **spring→spring**: acc=82.0%±8.0% (10 seeds)

## ext_more_seeds

- **spring_vjepa2_continuous_25seeds**: acc=86.1%±6.5% PD=0.536±0.078 CS=0.008±0.017
- **spring_vjepa2_discrete_25seeds**: acc=85.8%±7.7% PD=0.727±0.056 CS=0.123±0.107
- **spring_vjepa2_raw_probe_25seeds**: acc=85.9%±6.5% PD=0.707±0.034 CS=0.001±0.005

## ext_stability

- **agreement_rate**: 100.0%±0.0%