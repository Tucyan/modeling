from pathlib import Path
import json,hashlib,subprocess,sys
r=Path.cwd();o=r/'outputs/q3_fix_integration';sys.path.insert(0,str(r/'src'))
result=subprocess.run([sys.executable,'-X','utf8','-m','unittest','discover','-s','tests','-p','test_*.py','-v'],capture_output=True,text=True,encoding='utf-8')
(o/'all_tests.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
print('\n'.join((result.stdout+result.stderr).splitlines()[-5:]),flush=True)
if result.returncode:raise SystemExit(result.returncode)
from q3_experiments import load_inputs,simulate_policy,RiskPair
x=load_inputs();records=[]
for name,hours,expected in [('R0',(),393688.06172313704),('Rall',(6,12,18),366691.4185161416)]:
 daily,d,versions=simulate_policy(x,hours,RiskPair(.85,.75),start=31,end=38)
 cost=float(daily.cost.sum());assert abs(cost-expected)<1e-6,(name,cost,expected)
 record=dict(policy=name,cost=cost,emergency_kwh=float(d.e.sum()),bus_max=float(abs(d.q-d.unused+d.e+d.pv-d.spill+d.d-d.load-d.c).max()),state_max=float(abs(d.E_end-d.E_start-.9*d.c+d.d/.9).max()),continuity_max=float(abs(d.E_start.values[1:]-d.E_end.values[:-1]).max()),initial=float(d.E_start.iloc[0]),terminal=float(d.E_end.iloc[-1]),versions=len(versions))
 assert max(record[k] for k in ('bus_max','state_max','continuity_max'))<1e-6
 records.append(record);print(json.dumps(record),flush=True)
protected=json.loads((o/'protected_before.json').read_text(encoding='utf-8'));changed=[name for name,h in protected.items() if hashlib.sha256((r/name).read_bytes()).hexdigest()!=h];assert not changed,changed
archive=r/'outputs/q3_legacy_before_fix/outputs';s=r/'outputs/q3_zip_review/source/outputs';archive_changes=[];count=0
for p in archive.rglob('*'):
 if p.is_file():
  count+=1
  if hashlib.sha256(p.read_bytes()).hexdigest()!=hashlib.sha256((s/p.relative_to(archive)).read_bytes()).hexdigest():archive_changes.append(str(p))
assert not archive_changes,archive_changes
report=dict(test_exit_code=result.returncode,smoke=records,protected_files=len(protected),protected_changed=changed,archive_files=count,archive_changed=archive_changes,formal_rerun=False)
(o/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print('Protected files and archive hashes unchanged')
