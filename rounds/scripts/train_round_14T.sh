export HF_HOME=~/hf PYTHONWARNINGS=ignore TAG=_r15t RUN=_t14 CTXL=36 PROXY=style KREAD=1.0 OVERSAMPLE=0 MASK=1 MEANLOSS=1 CLOSS=1 CLOSS_SEASON=1 CLOSSW=0.5 DROP=1 NEWCOL=1 TYPEFILL=1 
cd ~/lead
cp t0lora_lead.r15.py t0lora_lead.py
(STEPS=2 RUN=_smoke_t14 ~/venv/bin/python t0lora_lead.py train --arm sonly && rm -rf adapter_sonly_v2t_smoke_t14 && cp -r adapter_sonly_r15t_smoke_t14 adapter_sonly_v2t_smoke_t14 && TAG=_v2t RUN=_smoke_t14 EVAL_LIMIT=64 EVAL_TAG=_smoke ~/venv/bin/python t0lora_lead.py eval --arm sonly) > log_smoke_t14.txt 2>&1 || { touch done_t14 failed_t14; exit 1; }
(STEPS=1500 ~/venv/bin/python t0lora_lead.py train --arm sonly && for t in v2t w14t; do rm -rf adapter_sonly_${t}_t14; cp -r adapter_sonly_r15t_t14 adapter_sonly_${t}_t14; done && TAG=_w14t ~/venv/bin/python t0lora_lead.py eval --arm sonly && TAG=_v2t ~/venv/bin/python t0lora_lead.py eval --arm sonly) > log_t14.txt 2>&1 || touch failed_t14
touch done_t14
