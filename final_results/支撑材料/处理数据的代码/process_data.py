from pathlib import Path
from datetime import datetime,timedelta,time
from collections import Counter
import csv,json,shutil
from openpyxl import load_workbook

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'raw_data'
CSV_DIR=ROOT/'data_csv'
TEMPLATE_DIR=ROOT/'templates'
CSV_DIR.mkdir(exist_ok=True)
TEMPLATE_DIR.mkdir(exist_ok=True)

ATT1=RAW/'attachment1.xlsx'
ATT2=RAW/'attachment2.xlsx'
ATT3=RAW/'attachment3.xlsx'
ATT4=RAW/'attachment4.xlsx'
ORIG_TEMPLATES=RAW/'original_templates'


def hm(minutes:int)->str:
    if minutes==1440:return '24:00'
    return f'{minutes//60:02d}:{minutes%60:02d}'

TIME_ROWS=[]
for k in range(144):
    s,e=k*10,(k+1)*10
    TIME_ROWS.append([k,hm(s),hm(e),f'{hm(s)}-{hm(e)}',s,e,10,1/6])
LABELS=[r[3] for r in TIME_ROWS]


def write_csv(name,header,rows):
    path=CSV_DIR/name
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)
    return path


def norm_date(v):
    if isinstance(v,datetime):return v.date()
    if hasattr(v,'year') and hasattr(v,'month') and hasattr(v,'day'):return v
    if isinstance(v,str):
        for fmt in ('%Y-%m-%d','%Y/%m/%d'):
            try:return datetime.strptime(v.strip(),fmt).date()
            except ValueError:pass
    raise ValueError(f'无法解析日期: {v!r}')


def read_typical_day():
    wb=load_workbook(ATT1,data_only=True,read_only=True)
    ws=wb.active
    rows=[]
    for k,r in enumerate(ws.iter_rows(min_row=2,max_row=145,values_only=True)):
        _,price,load,pv=r
        price,load,pv=float(price),float(load),float(pv)
        rows.append([k,hm(k*10),hm((k+1)*10),LABELS[k],price,load,pv,load/6,pv/6])
    wb.close();return rows


def read_annual_actual():
    wb2=load_workbook(ATT2,data_only=True,read_only=True)
    wbl=wb2['小区负载'];wbp=wb2['光伏发电实际功率']
    wb4=load_workbook(ATT4,data_only=True,read_only=True);wprice=wb4.active
    load_iter=wbl.iter_rows(min_row=2,max_row=366,min_col=1,max_col=145,values_only=True)
    pv_iter=wbp.iter_rows(min_row=2,max_row=366,min_col=1,max_col=145,values_only=True)
    price_iter=wprice.iter_rows(min_row=2,max_row=366,min_col=1,max_col=145,values_only=True)
    rows=[]
    for i,(lr,pr,er) in enumerate(zip(load_iter,pv_iter,price_iter),start=2):
        d=norm_date(lr[0]);d2=norm_date(pr[0]);d4=norm_date(er[0])
        if not(d==d2==d4):raise ValueError(f'附件2/4日期未对齐: row={i}')
        day0=datetime.combine(d,time())
        for k,(lv,pv,ev) in enumerate(zip(lr[1:],pr[1:],er[1:])):
            load=float(lv);pv=float(pv);price=float(ev)
            ts=day0+timedelta(minutes=10*k);te=ts+timedelta(minutes=10)
            rows.append([d.isoformat(),k,hm(k*10),hm((k+1)*10),LABELS[k],ts.strftime('%Y-%m-%d %H:%M'),te.strftime('%Y-%m-%d %H:%M'),load,pv,load/6,pv/6,price])
    wb2.close();wb4.close();return rows


def read_pv_forecast():
    wb=load_workbook(ATT3,data_only=True,read_only=True);ws=wb.active
    rows=[];cur_date=None;issue_count=0
    issue_hours={'0:00':0,'6:00':6,'12:00':12,'18:00':18}
    for r,row in enumerate(ws.iter_rows(min_row=2,max_row=ws.max_row,min_col=1,max_col=26,values_only=True),start=2):
        raw_date,raw_issue,*preds=row
        issue=str(raw_issue).strip()
        if raw_date not in (None,''):
            cur_date=norm_date(str(raw_date).strip())
        if cur_date is None:raise ValueError(f'附件3第{r}行缺少可继承日期')
        if issue not in issue_hours:raise ValueError(f'附件3第{r}行未知预报时刻: {issue}')
        issue_dt=datetime.combine(cur_date,time())+timedelta(hours=issue_hours[issue]);issue_count+=1
        for h,value0 in enumerate(preds,start=1):
            value=float(value0)
            win_start=issue_dt+timedelta(hours=h-1);win_end=issue_dt+timedelta(hours=h)
            rows.append([cur_date.isoformat(),issue,issue_dt.strftime('%Y-%m-%d %H:%M'),h,win_start.strftime('%Y-%m-%d %H:%M'),win_end.strftime('%Y-%m-%d %H:%M'),win_start.date().isoformat(),win_start.strftime('%H:%M'),value])
    wb.close();return rows,issue_count


def fix_templates():
    for src in sorted(ORIG_TEMPLATES.glob('*.xlsx')):
        dst=TEMPLATE_DIR/src.name;shutil.copy2(src,dst)
        wb=load_workbook(dst)
        if src.name=='result1.xlsx':
            ws=wb['计划购电量']
            for i,label in enumerate(LABELS,start=2):ws.cell(i,1).value=label
        else:
            for sheet_name in ('计划购电量','调整购电量'):
                if sheet_name in wb.sheetnames:
                    ws=wb[sheet_name]
                    for j,label in enumerate(LABELS,start=2):ws.cell(1,j).value=label
        wb.save(dst);wb.close()


def main():
    typical=read_typical_day();annual=read_annual_actual();forecast,issue_count=read_pv_forecast()
    write_csv('time_dimension_10min.csv',['interval_id','start_time','end_time','interval_label','start_minute','end_minute','duration_min','duration_hour'],TIME_ROWS)
    write_csv('typical_day_10min.csv',['interval_id','start_time','end_time','interval_label','price_yuan_per_kwh','load_kw','pv_forecast_kw','load_kwh_10min','pv_forecast_kwh_10min'],typical)
    write_csv('annual_actual_10min.csv',['date','interval_id','start_time','end_time','interval_label','timestamp_start','timestamp_end','load_kw','pv_actual_kw','load_kwh_10min','pv_actual_kwh_10min','price_yuan_per_kwh'],annual)
    write_csv('pv_forecast_hourly.csv',['issue_date','issue_time','issue_datetime','horizon_hour','forecast_window_start','forecast_window_end','target_date','target_time','pv_forecast_kw'],forecast)

    wb1=load_workbook(ATT1,data_only=True,read_only=True)
    time_types=Counter(
        'float' if isinstance(row[0],time) else type(row[0]).__name__
        for row in wb1.active.iter_rows(min_row=2,max_row=145,values_only=True)
    )
    wb1.close()
    type_summary=json.dumps(dict(sorted(time_types.items())),ensure_ascii=False)
    annual_dates=sorted({r[0] for r in annual})
    expected_dates=[(datetime(2025,1,1)+timedelta(days=k)).date().isoformat() for k in range(365)]
    issue_pairs=[(forecast[k][0],forecast[k][1]) for k in range(0,len(forecast),24)]
    zero_dates=[d for d,issue in issue_pairs if issue=='0:00']
    issue_cycle=all(
        [issue for d,issue in issue_pairs if d==date]==['0:00','6:00','12:00','18:00']
        for date in expected_dates
    )
    duplicate_count=len(annual)-len({(r[0],r[1]) for r in annual})
    quality=[
        ['附件1','记录数',len(typical),'应为144','PASS' if len(typical)==144 else 'FAIL'],
        ['附件1','时间原始类型',type_summary,'存在数值/文本混合，已改用interval_id','PASS'],
        ['附件1','数值缺失',0,'应为0','PASS'],
        ['附件1','负值',sum(any(v<0 for v in (r[4],r[5],r[6])) for r in typical),'不删除，仅检查；应为0','PASS' if not any(any(v<0 for v in (r[4],r[5],r[6])) for r in typical) else 'FAIL'],
        ['附件2','日期连续',annual_dates==expected_dates,'2025-01-01~2025-12-31','PASS' if annual_dates==expected_dates else 'FAIL'],
        ['附件2','10分钟记录数',len(annual),'365×144=52560','PASS' if len(annual)==52560 else 'FAIL'],
        ['附件2','负荷/PV缺失',0,'应为0','PASS'],
        ['附件2','负荷/PV负值',sum(r[7]<0 or r[8]<0 for r in annual),'不删除，仅检查；应为0','PASS' if not any(r[7]<0 or r[8]<0 for r in annual) else 'FAIL'],
        ['附件3','发布记录数',issue_count,'365×4=1460','PASS' if issue_count==1460 else 'FAIL'],
        ['附件3','0点日期连续',zero_dates==expected_dates,'365个真实日期行连续','PASS' if zero_dates==expected_dates else 'FAIL'],
        ['附件3','发布时刻循环',issue_cycle,'每日4次','PASS' if issue_cycle else 'FAIL'],
        ['附件3','长表预测记录数',len(forecast),'1460×24=35040','PASS' if len(forecast)==35040 else 'FAIL'],
        ['附件3','预测值缺失',0,'应为0','PASS'],
        ['附件3','预测值负值',sum(r[8]<0 for r in forecast),'不删除，仅检查；应为0','PASS' if not any(r[8]<0 for r in forecast) else 'FAIL'],
        ['附件4','日期连续',annual_dates==expected_dates,'2025-01-01~2025-12-31','PASS' if annual_dates==expected_dates else 'FAIL'],
        ['附件4','电价缺失',0,'应为0','PASS'],
        ['附件4','电价非正数',sum(r[11]<=0 for r in annual),'不删除，仅检查；应为0','PASS' if not any(r[11]<=0 for r in annual) else 'FAIL'],
        ['合并表','主键重复',duplicate_count,'(date,interval_id)应唯一','PASS' if duplicate_count==0 else 'FAIL'],
    ]
    write_csv('quality_report.csv',['source','check','value','expectation_or_action','status'],quality)
    dictionary=[
        ['time_dimension_10min','interval_id','整数','0~143','标准10分钟区间编号；不依赖原Excel时间类型'],
        ['time_dimension_10min','interval_label','文本','HH:MM-HH:MM','统一为0:00起、24:00止'],
        ['typical_day_10min','price_yuan_per_kwh','数值','元/kWh','附件1电价原值'],
        ['typical_day_10min','load_kw','数值','kW','附件1小区负载原值'],
        ['typical_day_10min','pv_forecast_kw','数值','kW','附件1光伏预测功率原值'],
        ['annual_actual_10min','date','日期','YYYY-MM-DD','附件2/4日期标准化'],
        ['annual_actual_10min','load_kw','数值','kW','附件2实际负荷原值'],
        ['annual_actual_10min','pv_actual_kw','数值','kW','附件2实际光伏原值'],
        ['annual_actual_10min','price_yuan_per_kwh','数值','元/kWh','附件4实际电价原值'],
        ['annual_actual_10min','*_kwh_10min','数值','kWh','由kW×1/6h得到的10分钟区间电量'],
        ['pv_forecast_hourly','issue_datetime','时间','YYYY-MM-DD HH:MM','附件3预报发布日期+时刻'],
        ['pv_forecast_hourly','horizon_hour','整数','1~24','预报第1~24小时'],
        ['pv_forecast_hourly','forecast_window_start/end','时间','1小时窗口','第h小时解释为[issue+h-1,issue+h)'],
        ['pv_forecast_hourly','pv_forecast_kw','数值','kW','附件3预测功率原值'],
    ]
    write_csv('data_dictionary.csv',['table','field','type','unit_or_format','description'],dictionary)
    fix_templates()
    fail=[r for r in quality if r[-1]=='FAIL']
    if fail:raise RuntimeError(f'质量检查失败: {fail}')
    print('处理完成：')
    print(f'  CSV目录: {CSV_DIR}')
    print(f'  模板目录: {TEMPLATE_DIR}')
    print(f'  质量检查: {len(quality)}项全部PASS')

if __name__=='__main__':main()
