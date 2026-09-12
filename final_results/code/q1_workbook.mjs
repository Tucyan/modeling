import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const output=path.join(root,'outputs/01a08b78-ed6b-7e60-b40d-9f266855d1ef/q1');
await fs.mkdir(output,{recursive:true});
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(root,'C题_标准化数据与修正模板/C_final_delivery/templates/result1.xlsx')));
if(process.argv.includes('--preview-template')){
  console.log((await wb.inspect({kind:'workbook,sheet,table',maxChars:2500,tableMaxRows:3,tableMaxCols:5})).ndjson);
  for(const [sheet,range,name] of [['计划购电量','A1:B12','template_plan'],['充放电量','A1:E7','template_storage']]){
    const b=await wb.render({sheetName:sheet,range,scale:1.5,format:'png'});
    await fs.writeFile(path.join(output,name+'.png'),new Uint8Array(await b.arrayBuffer()));
  }
}else{
  const solution=JSON.parse(await fs.readFile(path.join(output,'solution.json'),'utf8'));
  const plan=wb.worksheets.getItem('计划购电量');
  const storage=wb.worksheets.getItem('充放电量');
  plan.getRange('B2:B145').values=solution.purchase.map(v=>[v]);
  storage.getRange('B2:C7').values=solution.blocks.map(b=>[b.charge,b.discharge]);
  storage.getRange('E2:E3').values=[[solution.energy[0]],[solution.energy[144]]];
  plan.getRange('B2:B145').setNumberFormat('0.000000');
  storage.getRange('B2:C7').setNumberFormat('0.000000');
  storage.getRange('E2:E3').setNumberFormat('0.00');
  wb.recalculate();
  console.log((await wb.inspect({kind:'table',range:"'充放电量'!A1:E7",include:'values,formulas',tableMaxRows:7,tableMaxCols:5})).ndjson);
  console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#N/A',options:{useRegex:true,maxResults:20}})).ndjson);
  for(const [sheet,range,name] of [['计划购电量','A1:B16','result_plan'],['充放电量','A1:E7','result_storage']]){
    const b=await wb.render({sheetName:sheet,range,scale:1.5,format:'png'});
    await fs.writeFile(path.join(output,name+'.png'),new Uint8Array(await b.arrayBuffer()));
  }
  await (await SpreadsheetFile.exportXlsx(wb)).save(path.join(output,'result1.xlsx'));
  console.log('Saved result1.xlsx');
}
