import fs from "node:fs/promises";
import path from "node:path";
import {fileURLToPath} from "node:url";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const templatePath = path.join(root, "C题_标准化数据与修正模板", "C_final_delivery", "templates", "result3.xlsx");
const outputArg = process.argv.find((value) => value.startsWith("--output="));
const outputDir = outputArg ? path.resolve(root, outputArg.slice("--output=".length))
                            : path.join(root, "outputs", "q3_final");
const previewDir = path.join(root, "outputs", "q3_modeling", "workbook_previews");
const expectedSheets = ["计划购电量", "调整购电量", "充放电量", "紧急购电量"];

async function loadTemplate() {
  const input = await FileBlob.load(templatePath);
  const workbook = await SpreadsheetFile.importXlsx(input);
  const sheetInfo = await workbook.inspect({kind: "sheet", include: "id,name", maxChars: 4000});
  const found = workbook.worksheets.items.map((sheet) => sheet.name);
  if (JSON.stringify(found) !== JSON.stringify(expectedSheets)) {
    throw new Error(`Unexpected result3 sheets: ${JSON.stringify(found)}`);
  }
  return {workbook, sheetInfo};
}

async function renderAll(workbook, prefix) {
  await fs.mkdir(previewDir, {recursive: true});
  const ranges = ["A1:H8", "A1:H8", "A1:F14", "A1:C20"];
  for (let i = 0; i < expectedSheets.length; i += 1) {
    const sheetName = expectedSheets[i];
    const preview = await workbook.render({sheetName, range: ranges[i], scale: 1.3, format: "png"});
    await fs.writeFile(path.join(previewDir, `${prefix}_${i + 1}.png`),
                       new Uint8Array(await preview.arrayBuffer()));
  }
}

async function inspectTemplate() {
  const {workbook, sheetInfo} = await loadTemplate();
  const overview = await workbook.inspect({
    kind: "workbook,sheet,table", maxChars: 7000,
    tableMaxRows: 6, tableMaxCols: 8, tableMaxCellChars: 80,
  });
  process.stdout.write(`${sheetInfo.ndjson}\n${overview.ndjson}\n`);
  await renderAll(workbook, "template");
}

function writeBattery(sheet, battery) {
  if (battery.length !== 334 * 6) throw new Error("Expected 2004 battery rows");
  sheet.getRange("A2:F2005").clear({applyTo: "contents"});
  const rows = battery.map((row) => row.map(
    (value, column) => column === 0 && value ? new Date(`${value}T00:00:00Z`) : value));
  sheet.getRange("A2:F2005").values = rows;
  sheet.getRange("A2:A2005").setNumberFormat("yyyy/mm/dd");
  sheet.getRange("C2:D2005").setNumberFormat("0.000000");
  sheet.getRange("F2:F2005").setNumberFormat("0.000000");
  sheet.getRange("A1:F2005").format.columnWidth = 18;
  sheet.getRange("A1:F2005").format = {
    borders: {preset: "all", style: "thin", color: "#666666"},
    horizontalAlignment: "center", verticalAlignment: "center",
  };
}

function writeEmergency(sheet, emergency) {
  if (emergency.length < 334) throw new Error("Every day needs an emergency row");
  sheet.getRange("A2:C1000").clear({applyTo: "contents"});
  const rows = emergency.map((row) => row.map(
    (value, column) => column === 0 && value ? new Date(`${value}T00:00:00Z`) : value));
  sheet.getRangeByIndexes(1, 0, rows.length, 3).values = rows;
  sheet.getRangeByIndexes(1, 0, rows.length, 1).setNumberFormat("yyyy/mm/dd");
  sheet.getRangeByIndexes(1, 2, rows.length, 1).setNumberFormat("0.000000");
  sheet.getRangeByIndexes(0, 0, rows.length + 1, 3).format.columnWidth = 20;
  sheet.getRangeByIndexes(0, 0, rows.length + 1, 3).format = {
    borders: {preset: "all", style: "thin", color: "#666666"},
    horizontalAlignment: "center", verticalAlignment: "center",
  };
}

async function buildWorkbook() {
  const payload = JSON.parse(await fs.readFile(path.join(outputDir, "workbook_payload.json"), "utf8"));
  if (payload.plan.length !== 334 || payload.adjustment.length !== 334 ||
      payload.plan.some((row) => row.length !== 146) ||
      payload.adjustment.some((row) => row.length !== 146)) {
    throw new Error("Plan and adjustment matrices must be 334 by 146 including totals");
  }
  const {workbook} = await loadTemplate();
  workbook.worksheets.getItem("计划购电量").getRange("B2:EQ335").values = payload.plan;
  workbook.worksheets.getItem("调整购电量").getRange("B2:EQ335").values = payload.adjustment;
  workbook.worksheets.getItem("计划购电量").getRange("B2:EQ335").setNumberFormat("0.000000");
  workbook.worksheets.getItem("调整购电量").getRange("B2:EQ335").setNumberFormat("0.000000");
  writeBattery(workbook.worksheets.getItem("充放电量"), payload.battery);
  writeEmergency(workbook.worksheets.getItem("紧急购电量"), payload.emergency);
  workbook.recalculate();

  const planCheck = await workbook.inspect({
    kind: "table", range: "计划购电量!A1:H6", include: "values,formulas",
    tableMaxRows: 6, tableMaxCols: 8, maxChars: 5000,
  });
  const storageCheck = await workbook.inspect({
    kind: "table", range: "充放电量!A1:F8", include: "values,formulas",
    tableMaxRows: 8, tableMaxCols: 6, maxChars: 5000,
  });
  const errors = await workbook.inspect({
    kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: {useRegex: true, maxResults: 300}, summary: "final formula error scan",
  });
  process.stdout.write(`${planCheck.ndjson}\n${storageCheck.ndjson}\n${errors.ndjson}\n`);
  await renderAll(workbook, `final_${path.basename(outputDir)}`);
  await fs.mkdir(outputDir, {recursive: true});
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(path.join(outputDir, "result3.xlsx"));
}

if (process.argv.includes("--inspect-template")) await inspectTemplate();
else if (process.argv.includes("--help-table-resize")) {
  const {workbook} = await loadTemplate();
  process.stdout.write(workbook.help("table.resize", {include: "index,examples,notes", maxChars: 4000}).ndjson);
} else await buildWorkbook();
