import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const outputFlag=process.argv.indexOf('--out');
const hasV2=await fs.access(path.join(root,'outputs/q2_review_v2/selection_protocol.json')).then(()=>true,()=>false);
const out=outputFlag>=0?path.resolve(root,process.argv[outputFlag+1]):path.join(root,hasV2?'outputs/q2_review_v2':'outputs/01a08b78-ed6b-7e60-b40d-9f266855d1ef/q2');
await fs.mkdir(out,{recursive:true});
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(root,'C题_标准化数据与修正模板/C_final_delivery/templates/result2.xlsx')));
if(process.argv.includes('--preview-template')){
  for(const [sheet,range] of [['计划购电量','A1:F6'],['充放电量','A1:F9'],['紧急购电量','A1:C11']]){
    const blob=await wb.render({sheetName:sheet,range,scale:1.3,format:'png'});
    await fs.writeFile(path.join(out,'template_'+sheet+'.png'),new Uint8Array(await blob.arrayBuffer()));
  }
}else{
  const a=JSON.parse(await fs.readFile(path.join(out,'workbook_payload.json'),'utf8'));
  const plan=wb.worksheets.getItem('计划购电量'), battery=wb.worksheets.getItem('充放电量'), emergency=wb.worksheets.getItem('紧急购电量');
  plan.getRange('B2:EQ335').values=a.plan;
  plan.getRange('B2:EQ335').setNumberFormat('0.000000');
  plan.getRange('B1:EQ335').format.columnWidth=16;
  battery.getRange('A2:F20').clear({applyTo:'contents'});
  const batt=a.battery.map(r=>r.map((v,j)=>j===0&&v?new Date(v+'T00:00:00Z'):v));
  battery.getRange('A2:F2005').values=batt;
  battery.getRange('A2:A2005').setNumberFormat('yyyy/mm/dd');
  battery.getRange('C2:D2005').setNumberFormat('0.000000');
  battery.getRange('F2:F2005').setNumberFormat('0.000000');
  battery.getRange('A1:F2005').format.columnWidth=18;
  emergency.getRange('A2:C11').clear({applyTo:'contents'});
  const er=a.emergency.map(r=>r.map((v,j)=>j===0&&v?new Date(v+'T00:00:00Z'):v));
  emergency.getRangeByIndexes(1,0,er.length,3).values=er;
  emergency.getRangeByIndexes(1,0,er.length,1).setNumberFormat('yyyy/mm/dd');
  emergency.getRangeByIndexes(1,2,er.length,1).setNumberFormat('0.000000');
  emergency.getRangeByIndexes(0,0,er.length+1,3).format.columnWidth=20;
  wb.recalculate();
  console.log((await wb.inspect({kind:'table',range:"'充放电量'!A1:F8",tableMaxRows:8,tableMaxCols:6,maxChars:1500})).ndjson);
  console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NUM!',options:{useRegex:true,maxResults:10},maxChars:500})).ndjson);
  for(const [sheet,range] of [['计划购电量','A1:F8'],['充放电量','A1:F9'],['紧急购电量','A1:C12']]){
    const blob=await wb.render({sheetName:sheet,range,scale:1.3,format:'png'});
    await fs.writeFile(path.join(out,'result_'+sheet+'.png'),new Uint8Array(await blob.arrayBuffer()));
  }
  await(await SpreadsheetFile.exportXlsx(wb)).save(path.join(out,'result2.xlsx'));
  console.log('Saved result2.xlsx');
}
