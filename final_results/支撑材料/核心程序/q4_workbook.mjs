import fs from "node:fs/promises";
import path from "node:path";
import {fileURLToPath} from "node:url";
import {FileBlob, SpreadsheetFile} from "@oai/artifact-tool";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outputArg = process.argv.find((value) => value.startsWith("--output="));
const modeArg = process.argv.find((value) => value.startsWith("--mode="));
const outputDir = outputArg ? path.resolve(root, outputArg.slice("--output=".length))
                            : path.join(root, "outputs", "q4_final", "N0");
const mode = modeArg ? modeArg.slice("--mode=".length) : "q4-3";
if (!["q4-2", "q4-3"].includes(mode)) throw new Error("mode must be q4-2 or q4-3");
const templateName = mode === "q4-2" ? "result4-2.xlsx" : "result4-3.xlsx";
const templatePath = path.join(root, "C题_标准化数据与修正模板", "C_final_delivery",
                               "templates", templateName);
const sheets = mode === "q4-2"
  ? ["计划购电量", "充放电量", "紧急购电量"]
  : ["计划购电量", "调整购电量", "充放电量", "紧急购电量"];

async function loadTemplate() {
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(templatePath));
  const found = workbook.worksheets.items.map((sheet) => sheet.name);
  if (JSON.stringify(found) !== JSON.stringify(sheets))
    throw new Error(`Unexpected ${templateName} sheets: ${JSON.stringify(found)}`);
  return workbook;
}

function writeBattery(sheet, rows) {
  if (rows.length !== 334 * 6) throw new Error("Expected 2004 battery rows");
  sheet.getRange("A2:F2005").clear({applyTo: "contents"});
  sheet.getRange("A2:F2005").values = rows.map((row) => row.map(
    (value, column) => column === 0 && value ? new Date(`${value}T00:00:00Z`) : value));
  sheet.getRange("A2:A2005").setNumberFormat("yyyy/mm/dd");
  sheet.getRange("C2:D2005").setNumberFormat("0.000000");
  sheet.getRange("F2:F2005").setNumberFormat("0.000000");
}

function writeEmergency(sheet, rows) {
  if (rows.length < 334) throw new Error("Every day needs an emergency row");
  sheet.getRange("A2:C1000").clear({applyTo: "contents"});
  sheet.getRangeByIndexes(1, 0, rows.length, 3).values = rows.map((row) => row.map(
    (value, column) => column === 0 && value ? new Date(`${value}T00:00:00Z`) : value));
  sheet.getRangeByIndexes(1, 0, rows.length, 1).setNumberFormat("yyyy/mm/dd");
  sheet.getRangeByIndexes(1, 2, rows.length, 1).setNumberFormat("0.000000");
}

const payload = JSON.parse(await fs.readFile(path.join(outputDir, "workbook_payload.json"), "utf8"));
if (payload.plan.length !== 334 || payload.plan.some((row) => row.length !== 146))
  throw new Error("Plan matrix must be 334 by 146 including totals");
const workbook = await loadTemplate();
workbook.worksheets.getItem("计划购电量").getRange("B2:EQ335").values = payload.plan;
workbook.worksheets.getItem("计划购电量").getRange("B2:EQ335").setNumberFormat("0.000000");
if (mode === "q4-3") {
  if (payload.adjustment.length !== 334 || payload.adjustment.some((row) => row.length !== 146))
    throw new Error("Adjustment matrix must be 334 by 146 including totals");
  workbook.worksheets.getItem("调整购电量").getRange("B2:EQ335").values = payload.adjustment;
  workbook.worksheets.getItem("调整购电量").getRange("B2:EQ335").setNumberFormat("0.000000");
}
writeBattery(workbook.worksheets.getItem("充放电量"), payload.battery);
writeEmergency(workbook.worksheets.getItem("紧急购电量"), payload.emergency);
workbook.recalculate();
const errors = await workbook.inspect({
  kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: {useRegex: true, maxResults: 300}, summary: "Q4 formula error scan",
});
process.stdout.write(`${errors.ndjson}\n`);
const previews = path.join(outputDir, "workbook_previews");
await fs.mkdir(previews, {recursive: true});
const ranges = mode === "q4-2" ? ["A1:H8", "A1:F8", "A1:C12"]
                               : ["A1:H8", "A1:H8", "A1:F8", "A1:C12"];
for (let i = 0; i < sheets.length; i += 1) {
  const image = await workbook.render({sheetName: sheets[i], range: ranges[i], scale: 1.3, format: "png"});
  await fs.writeFile(path.join(previews, `final_${i + 1}.png`),
                     new Uint8Array(await image.arrayBuffer()));
}
await fs.mkdir(outputDir, {recursive: true});
await (await SpreadsheetFile.exportXlsx(workbook)).save(path.join(outputDir, templateName));
console.log(`Saved ${templateName}`);
