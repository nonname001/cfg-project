#!/bin/bash
# Second pass: MLX memory/timing with load buffers freed, repeated full-length XL/2, reference runs.
W=$1; PY=.venv/bin/python; R=bench/results_pass2.jsonl; : > $R
therm() { echo "THERM $1 $(pmset -g therm | grep -i -E 'speed_limit|thermal warning|performance warning' | tr '\n' ' ')" | tee -a bench/therm.log; }
for s in S B L XL; do for p in 8 4 2; do
  m=SiT-$s/$p; t=SiT-${s}_$p
  [ -f $W/$t.safetensors ] || $PY bench/prep.py $m $W/$t 0 >/dev/null 2>&1
  [ "$m" = SiT-XL/2 ] && continue
  for c in "" compile; do $PY bench/worker.py mlx $m $W/$t $W/$t.p2.npy 50 $c 2>/dev/null | grep RESULT | cut -c8- | tee -a $R; sleep 10; done
  rm -f $W/$t.safetensors
done; done
therm before_xl2; sleep 120
t=SiT-XL_2; m=SiT-XL/2
for round in 1 2; do
  for c in "" compile; do $PY bench/worker.py mlx $m $W/$t $W/$t.p2.npy 50 $c 2>/dev/null | grep RESULT | cut -c8- | tee -a $R; therm r$round$c; sleep 90; done
  $PY bench/worker.py mps $m $W/$t $W/$t.mps2.npy 50 2>/dev/null | grep RESULT | cut -c8- | tee -a $R; therm r${round}mps; sleep 90
done
rm -f $W/$t.safetensors
echo "=== dopri5 on MPS"
$PY - <<'PY' 2>&1 | grep -v -i warn | tail -2 | cut -c1-200
import sys; sys.path.insert(0,'sit')
import torch
from models import SiT_models
from transport import create_transport, Sampler
torch.set_grad_enabled(False)
m=SiT_models['SiT-S/8'](input_size=32,num_classes=1000,learn_sigma=True).to('mps').eval()
fn=Sampler(create_transport('Linear','velocity',None,None,None)).sample_ode(sampling_method='dopri5',num_steps=50)
z=torch.randn(2,4,32,32,device='mps'); y=torch.tensor([1,1000],device='mps')
print('dopri5 mps ok', fn(z,m.forward_with_cfg,y=y,cfg_scale=4.0)[-1].shape)
PY
echo "=== unmodified sample.py (CPU), euler 50"
/usr/bin/time -l $PY sit/sample.py ODE --sampling-method euler --num-sampling-steps 50 --seed 0 2>&1 | grep -E "Sampling took|real|peak memory footprint|maximum resident|Error" 
mv sample.png outputs/SiT-XL_2_unmodified_sample_py_cpu.png
therm end
