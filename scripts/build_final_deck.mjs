import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { FileBlob, Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = process.env.DECK_WORKSPACE;
const skillDir = process.env.DECK_SKILL_DIR;
const runtimePython = process.env.DECK_RUNTIME_PYTHON;
if (!workspaceDir || !skillDir || !runtimePython) {
  throw new Error("DECK_WORKSPACE, DECK_SKILL_DIR and DECK_RUNTIME_PYTHON are required");
}

const {
  applyPresentationChartFont,
  finalizePresentation,
  makeNativeBulletParagraphs,
  resolvePresentationFont,
} = await import(pathToFileURL(path.join(skillDir, "container_tools", "artifact_tool_utils.mjs")).href);

const buildDir = path.join(workspaceDir, "build", "deck");
const renderDir = path.join(buildDir, "rendered");
const reviewDir = path.join(buildDir, "review");
const candidatePath = path.join(buildDir, "candidate.pptx");
const reviewPath = path.join(reviewDir, "traffic_volume_prediction_final_v3.pptx");
const FONT = resolvePresentationFont({ fontFamily: "Arial" });

const C = {
  navy: "#0B1F33",
  ink: "#142735",
  muted: "#5D6B78",
  paper: "#F7F4ED",
  white: "#FFFFFF",
  teal: "#118A8A",
  tealLight: "#D7EEEE",
  orange: "#E47B36",
  orangeLight: "#F7E0D0",
  blue: "#2C7FB8",
  red: "#C94845",
  line: "#D9DEE3",
  green: "#4F8A5B",
};

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

async function readCsv(relativePath) {
  return parseCsv(await fs.readFile(path.join(workspaceDir, relativePath), "utf8"));
}

async function readImageBlob(relativePath) {
  const bytes = await fs.readFile(path.join(workspaceDir, relativePath));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function addText(slide, text, position, style = {}, name = undefined) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: FONT,
    fontSize: "20pt",
    color: C.ink,
    autoFit: "shrinkText",
    ...style,
  };
  return shape;
}

function addRule(slide, top, left = 64, width = 1152, fill = C.line) {
  slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height: 2 },
    fill,
    line: { style: "solid", fill, width: 0 },
  });
}

function addHeader(slide, title, number) {
  addText(slide, title, { left: 64, top: 38, width: 1080, height: 52 }, {
    fontSize: "29pt",
    bold: true,
    color: C.navy,
  }, `title-${number}`);
  addText(slide, String(number).padStart(2, "0"), { left: 1150, top: 47, width: 66, height: 30 }, {
    fontSize: "13pt",
    bold: true,
    color: C.teal,
    alignment: "right",
  });
  addRule(slide, 98, 64, 1152, C.teal);
}

function addFooter(slide) {
  addText(slide, "Traffic volume prediction", { left: 64, top: 684, width: 260, height: 20 }, {
    fontSize: "9pt",
    color: C.muted,
  });
}

function addBullets(slide, items, position, fontSize = "18pt", color = C.ink) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = makeNativeBulletParagraphs(items, {
    marginLeftPoints: 18,
    hangingPoints: 9,
    spaceAfterPoints: 8,
  });
  box.text.style = {
    typeface: FONT,
    fontSize,
    color,
    autoFit: "shrinkText",
  };
  return box;
}

function styleTable(table, headerFill = C.navy) {
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: table.columns.length }).assign({
    fill: headerFill,
    textStyle: { typeface: FONT, color: C.white, bold: true, fontSize: 15 },
    margins: { left: 8, right: 8, top: 6, bottom: 6 },
  });
  if (table.rows.length > 1) {
    table.cells.block({ row: 1, column: 0, rowCount: table.rows.length - 1, columnCount: table.columns.length }).assign({
      fill: C.white,
      textStyle: { typeface: FONT, color: C.ink, fontSize: 14 },
      margins: { left: 8, right: 8, top: 5, bottom: 5 },
    });
  }
}

function styleChart(chart) {
  applyPresentationChartFont(chart, { fontFamily: FONT });
  chart.chartFill = { color: C.paper, transparency: 100000 };
  chart.plotAreaFill = { color: C.white, transparency: 100000 };
  return chart;
}

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(renderDir, { recursive: true });
await fs.mkdir(reviewDir, { recursive: true });

const quality = await readCsv("reports/tables/data_quality_summary.csv");
const regression = await readCsv("results/metrics/regression_test.csv");
const ablation = await readCsv("results/metrics/ablation_validation.csv");
const classification = await readCsv("results/metrics/classification_test.csv");
const groupedImportance = await readCsv("reports/tables/grouped_feature_importance.csv");
const featureImportance = await readCsv("reports/tables/feature_importance.csv");

const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
presentation.theme.colors = {
  accent1: C.teal,
  accent2: C.orange,
  accent3: C.blue,
  accent4: C.green,
  accent5: C.red,
  accent6: C.muted,
  dark1: C.navy,
  light1: C.paper,
};

// 1. Cover
{
  const slide = presentation.slides.add();
  slide.background.fill = C.navy;
  slide.shapes.add({
    geometry: "rect",
    position: { left: 64, top: 88, width: 10, height: 470 },
    fill: C.orange,
    line: { style: "solid", fill: C.orange, width: 0 },
  });
  addText(slide, "Traffic volume prediction\nand congestion analysis", { left: 112, top: 118, width: 930, height: 220 }, {
    fontSize: "43pt",
    bold: true,
    color: C.white,
  });
  addText(slide, "Metro Interstate Traffic Volume", { left: 114, top: 366, width: 700, height: 46 }, {
    fontSize: "22pt",
    color: "#B9DCDC",
  });
  addText(slide, "Chronological forecasting, congestion labels, and error analysis", { left: 114, top: 426, width: 840, height: 54 }, {
    fontSize: "18pt",
    color: "#D8E0E7",
  });
  addText(slide, "Course project  |  2012 to 2018", { left: 114, top: 612, width: 520, height: 28 }, {
    fontSize: "13pt",
    color: "#9FB0BF",
  });
  slide.speakerNotes.text = "Dataset source: UCI Machine Learning Repository, Metro Interstate Traffic Volume, https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume";
}

// 2. Problem and dataset
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  addHeader(slide, "Problem and dataset", 2);
  addText(slide, "Forecast hourly traffic volume and classify each hour as low, moderate, or high congestion.", { left: 70, top: 132, width: 600, height: 88 }, {
    fontSize: "24pt",
    bold: true,
    color: C.navy,
  });
  addText(slide, "The data combines traffic counts with local weather observations on westbound I-94 near Minneapolis. Multiple weather descriptions can occur at the same timestamp, and the source contains long gaps.", { left: 70, top: 244, width: 590, height: 170 }, {
    fontSize: "18pt",
    color: C.ink,
  });
  addText(slide, "48,204", { left: 760, top: 150, width: 220, height: 68 }, { fontSize: "36pt", bold: true, color: C.teal });
  addText(slide, "raw weather rows", { left: 760, top: 216, width: 260, height: 34 }, { fontSize: "16pt", color: C.muted });
  addRule(slide, 276, 760, 400, C.line);
  addText(slide, "40,575", { left: 760, top: 302, width: 220, height: 68 }, { fontSize: "36pt", bold: true, color: C.orange });
  addText(slide, "unique observed hours", { left: 760, top: 368, width: 300, height: 34 }, { fontSize: "16pt", color: C.muted });
  addRule(slide, 428, 760, 400, C.line);
  addText(slide, "6 years", { left: 760, top: 454, width: 220, height: 68 }, { fontSize: "36pt", bold: true, color: C.blue });
  addText(slide, "October 2012 to September 2018", { left: 760, top: 520, width: 380, height: 34 }, { fontSize: "16pt", color: C.muted });
  addFooter(slide);
  slide.speakerNotes.text = "Dataset source: UCI Machine Learning Repository, Metro Interstate Traffic Volume, https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume";
}

// 3. Cleaning and quality
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  addHeader(slide, "Data cleaning and quality", 3);
  const selected = [
    "Raw rows",
    "Duplicate timestamp rows collapsed",
    "Hourly grid rows",
    "Missing hours added",
    "Zero Kelvin values replaced",
    "Extreme rain values replaced",
    "Traffic targets imputed",
  ].map((metric) => quality.find((row) => row.Metric === metric));
  const table = slide.tables.add({
    rows: selected.length + 1,
    columns: 2,
    left: 70,
    top: 135,
    width: 670,
    height: 435,
    columnWidths: [500, 170],
    values: [["Quality check", "Count"], ...selected.map((row) => [row.Metric, Number(row.Value).toLocaleString()])],
  });
  styleTable(table);
  addText(slide, "Cleaning decisions", { left: 790, top: 146, width: 340, height: 42 }, {
    fontSize: "22pt",
    bold: true,
    color: C.navy,
  });
  addBullets(slide, [
    "Collapse duplicate weather rows to one timestamp.",
    "Create a continuous hourly index while keeping missing traffic explicit.",
    "Forward-fill weather for at most six past hours.",
    "Expand midnight holiday labels across the calendar date.",
  ], { left: 790, top: 206, width: 400, height: 310 }, "17pt");
  addText(slide, "The longest source outage spans 7,386 hours. No traffic value is interpolated across it.", { left: 790, top: 540, width: 390, height: 74 }, {
    fontSize: "16pt",
    bold: true,
    color: C.red,
  });
  addFooter(slide);
  slide.speakerNotes.text = "Source: reports/tables/data_quality_summary.csv generated by src/clean.py.";
}

// 4. EDA
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  addHeader(slide, "Training-period traffic patterns", 4);
  slide.images.add({
    blob: await readImageBlob("reports/figures/01_hourly_weekday_vs_weekend.png"),
    contentType: "image/png",
    alt: "Weekday and weekend hourly traffic profiles",
    fit: "contain",
    position: { left: 54, top: 124, width: 590, height: 500 },
  });
  slide.images.add({
    blob: await readImageBlob("reports/figures/02_dow_hour_heatmap.png"),
    contentType: "image/png",
    alt: "Day of week by hour traffic heatmap",
    fit: "contain",
    position: { left: 642, top: 124, width: 590, height: 500 },
  });
  addText(slide, "Weekdays show clear morning and afternoon peaks. Weekend demand rises later and stays flatter through the day.", { left: 116, top: 622, width: 1040, height: 44 }, {
    fontSize: "16pt",
    bold: true,
    alignment: "center",
    color: C.navy,
  });
  addFooter(slide);
  slide.speakerNotes.text = "Figures generated from the training split by src/eda_temporal.py.";
}

// 5. Split and safeguards
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  addHeader(slide, "Chronological split and leakage controls", 5);
  const table = slide.tables.add({
    rows: 4,
    columns: 4,
    left: 70,
    top: 138,
    width: 1140,
    height: 220,
    columnWidths: [190, 360, 180, 220],
    values: [
      ["Split", "Date range", "Grid hours", "Eligible model rows"],
      ["Train", "2 Oct 2012 to 13 Dec 2016", "36,785", "5,621"],
      ["Validation", "13 Dec 2016 to 6 Nov 2017", "7,883", "5,701"],
      ["Test", "6 Nov 2017 to 30 Sep 2018", "7,883", "5,587"],
    ],
  });
  styleTable(table);
  addText(slide, "Safeguards", { left: 70, top: 402, width: 220, height: 38 }, {
    fontSize: "21pt",
    bold: true,
    color: C.navy,
  });
  addBullets(slide, [
    "TimeSeriesSplit preserves order and leaves a 24-hour gap between folds.",
    "Every rolling window starts after shift(1).",
    "Imputation, encoding, and scaling fit inside each training pipeline.",
    "All feature sets use the same timestamps for fair comparisons.",
    "Training quantiles define congestion thresholds once.",
  ], { left: 70, top: 454, width: 1090, height: 182 }, "17pt");
  addFooter(slide);
  slide.speakerNotes.text = "Split counts come from src/split.py and results/feature_sets.json.";
}

// 6. Features
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  addHeader(slide, "Feature design", 6);
  const table = slide.tables.add({
    rows: 4,
    columns: 3,
    left: 70,
    top: 140,
    width: 1140,
    height: 330,
    columnWidths: [190, 580, 370],
    values: [
      ["Group", "Features", "Purpose"],
      ["Temporal", "Hour, weekday, month, cyclical encodings, holiday, rush hour", "Capture regular commuting patterns"],
      ["Weather", "Temperature, rain, snow, cloud cover, condition, rush interactions", "Measure short-term environmental effects"],
      ["Lags", "1, 2, 24, 168 hours plus shifted rolling mean and standard deviation", "Represent recent traffic and weekly recurrence"],
    ],
  });
  styleTable(table);
  addText(slide, "Six ablation sets isolate each group and their combinations: T, W, L, T+W, T+L, and T+W+L.", { left: 105, top: 520, width: 1070, height: 70 }, {
    fontSize: "22pt",
    bold: true,
    alignment: "center",
    color: C.teal,
  });
  addFooter(slide);
  slide.speakerNotes.text = "Feature definitions: results/feature_sets.json and src/features/.";
}

// 7. Regression results
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  addHeader(slide, "Final regression results", 7);
  const rows = regression.filter((row) => ["XGBoost", "Random Forest (T+L selected)", "Random Forest", "PyTorch MLP", "Same hour last week"].includes(row.Model));
  const chart = slide.charts.add("bar", {
    position: { left: 60, top: 135, width: 650, height: 455 },
    categories: rows.map((row) => row.Model.replace("Random Forest (T+L selected)", "Random Forest T+L")),
    series: [{ name: "MAE", values: rows.map((row) => Number(Number(row.MAE).toFixed(1))), fill: C.teal }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 48 },
    hasLegend: false,
    xAxis: {
      title: { text: "Test MAE (vehicles/hour)", textStyle: { fontSize: 13, fill: C.muted } },
      min: 0,
      max: 400,
      majorGridlines: { style: "solid", fill: C.line, width: 1 },
      textStyle: { fontSize: 12, fill: C.muted },
    },
    yAxis: { textStyle: { fontSize: 12, fill: C.ink }, line: { style: "solid", fill: C.line, width: 1 } },
    dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0.0", textStyle: { fontSize: 12, bold: true, fill: C.navy } },
  });
  styleChart(chart);
  const tableRows = regression.slice(0, 5);
  const table = slide.tables.add({
    rows: tableRows.length + 1,
    columns: 4,
    left: 740,
    top: 164,
    width: 470,
    height: 345,
    columnWidths: [190, 80, 100, 100],
    values: [["Model", "MAE", "RMSE", "R²"], ...tableRows.map((row) => [
      row.Model.replace("Random Forest (T+L selected)", "RF T+L"),
      Number(row.MAE).toFixed(1),
      Number(row.RMSE).toFixed(1),
      Number(row.R2).toFixed(3),
    ])],
  });
  styleTable(table);
  addText(slide, "XGBoost and the selected T+L random forest finished within 0.01 MAE.", { left: 760, top: 548, width: 420, height: 64 }, {
    fontSize: "17pt",
    bold: true,
    color: C.orange,
    alignment: "center",
  });
  addFooter(slide);
  slide.speakerNotes.text = "Source: results/metrics/regression_test.csv. All models were refit on train plus validation before one test evaluation.";
}

// 8. Ablation
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  addHeader(slide, "Feature ablation", 8);
  const setOrder = ["T", "W", "L", "T+W", "T+L", "T+W+L"];
  const modelOrder = ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"];
  const seriesColors = [C.muted, C.orange, C.teal, C.blue];
  const chart = slide.charts.add("bar", {
    position: { left: 70, top: 130, width: 1120, height: 465 },
    categories: setOrder,
    series: modelOrder.map((model, index) => ({
      name: model,
      values: setOrder.map((set) => Number(Number(ablation.find((row) => row.Feature_Set === set && row.Model === model).MAE).toFixed(1))),
      fill: seriesColors[index],
    })),
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 48 },
    hasLegend: true,
    legend: { position: "bottom", overlay: false, textStyle: { fontSize: 12, fill: C.ink } },
    xAxis: { textStyle: { fontSize: 13, fill: C.ink }, line: { style: "solid", fill: C.line, width: 1 } },
    yAxis: {
      title: { text: "Validation MAE", textStyle: { fontSize: 13, fill: C.muted } },
      min: 0,
      max: 1700,
      majorGridlines: { style: "solid", fill: C.line, width: 1 },
      textStyle: { fontSize: 11, fill: C.muted },
    },
  });
  styleChart(chart);
  addText(slide, "Lag features produce the largest gain. Weather alone performs poorly, and T+L gives the best validation MAE.", { left: 120, top: 610, width: 1040, height: 50 }, {
    fontSize: "18pt",
    bold: true,
    alignment: "center",
    color: C.navy,
  });
  addFooter(slide);
  slide.speakerNotes.text = "Source: results/metrics/ablation_validation.csv. Hyperparameters stayed fixed across feature sets.";
}

// 9. Importance
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  addHeader(slide, "Feature importance", 9);
  const chart = slide.charts.add("bar", {
    position: { left: 70, top: 150, width: 520, height: 360 },
    categories: groupedImportance.map((row) => row.Group),
    series: [{ name: "Share", values: groupedImportance.map((row) => Number(Number(row.Share).toFixed(4))), fill: C.teal }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 55 },
    hasLegend: false,
    xAxis: {
      min: 0,
      max: 0.75,
      numberFormatCode: "0%",
      majorGridlines: { style: "solid", fill: C.line, width: 1 },
      textStyle: { fontSize: 12, fill: C.muted },
    },
    yAxis: { textStyle: { fontSize: 15, fill: C.ink } },
    dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0%", textStyle: { fontSize: 14, bold: true, fill: C.navy } },
  });
  styleChart(chart);
  addText(slide, "Highest permutation importance", { left: 690, top: 146, width: 430, height: 44 }, {
    fontSize: "22pt",
    bold: true,
    color: C.navy,
  });
  const topFeatures = featureImportance.slice(0, 6).map((row) => `${row.Feature}: +${Number(row.Permutation_MAE_Increase).toFixed(1)} MAE`);
  addBullets(slide, topFeatures, { left: 690, top: 210, width: 470, height: 300 }, "17pt");
  addText(slide, "Permutation importance uses the validation period. The final test remains reserved for evaluation and error analysis.", { left: 120, top: 570, width: 1040, height: 66 }, {
    fontSize: "16pt",
    bold: true,
    color: C.muted,
    alignment: "center",
  });
  addFooter(slide);
  slide.speakerNotes.text = "Source: reports/tables/feature_importance.csv and grouped_feature_importance.csv. Importance was computed on validation predictions.";
}

// 10. Classification
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  addHeader(slide, "Congestion classification", 10);
  const classRows = classification;
  const chart = slide.charts.add("bar", {
    position: { left: 60, top: 140, width: 650, height: 430 },
    categories: classRows.map((row) => row.Model),
    series: [{ name: "Macro-F1", values: classRows.map((row) => Number(Number(row.Macro_F1).toFixed(4))), fill: C.blue }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 48 },
    hasLegend: false,
    xAxis: {
      min: 0,
      max: 1,
      numberFormatCode: "0.0",
      majorGridlines: { style: "solid", fill: C.line, width: 1 },
      textStyle: { fontSize: 12, fill: C.muted },
    },
    yAxis: { textStyle: { fontSize: 12, fill: C.ink } },
    dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0.000", textStyle: { fontSize: 12, bold: true, fill: C.navy } },
  });
  styleChart(chart);
  const table = slide.tables.add({
    rows: 4,
    columns: 3,
    left: 760,
    top: 168,
    width: 420,
    height: 248,
    columnWidths: [140, 140, 140],
    values: [
      ["Class", "Lower bound", "Upper bound"],
      ["Low", "0", "2,228"],
      ["Moderate", "2,229", "4,510"],
      ["High", "4,511", "No limit"],
    ],
  });
  styleTable(table);
  addText(slide, "Random Forest led the final test with Macro-F1 0.9402 and balanced accuracy 0.9399.", { left: 770, top: 468, width: 400, height: 92 }, {
    fontSize: "19pt",
    bold: true,
    color: C.teal,
    alignment: "center",
  });
  addFooter(slide);
  slide.speakerNotes.text = "Source: results/metrics/classification_test.csv and results/thresholds.json. Thresholds were fixed from training quantiles.";
}

// 11. Error analysis
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  addHeader(slide, "Error analysis", 11);
  slide.images.add({
    blob: await readImageBlob("reports/figures/11_residual_analysis.png"),
    contentType: "image/png",
    alt: "Residual error by hour and weather condition",
    fit: "contain",
    position: { left: 42, top: 124, width: 620, height: 470 },
  });
  slide.images.add({
    blob: await readImageBlob("reports/figures/12_predicted_vs_actual.png"),
    contentType: "image/png",
    alt: "Predicted and actual traffic during a sample test week",
    fit: "contain",
    position: { left: 635, top: 124, width: 610, height: 470 },
  });
  addText(slide, "Errors vary by hour and weather condition, but the selected model follows the test-week commuting pattern closely.", { left: 120, top: 610, width: 1040, height: 48 }, {
    fontSize: "17pt",
    bold: true,
    alignment: "center",
    color: C.navy,
  });
  addFooter(slide);
  slide.speakerNotes.text = "Source: reports/tables/residuals_by_hour.csv, residuals_by_weather.csv, and results/predictions/regression_test.parquet.";
}

// 12. Conclusions and limitations
{
  const slide = presentation.slides.add();
  slide.background.fill = C.navy;
  addText(slide, "Conclusions and limitations", { left: 64, top: 46, width: 1000, height: 64 }, {
    fontSize: "31pt",
    bold: true,
    color: C.white,
  });
  addRule(slide, 118, 64, 1152, C.orange);
  addText(slide, "What worked", { left: 80, top: 160, width: 400, height: 44 }, {
    fontSize: "24pt",
    bold: true,
    color: "#A8DEDE",
  });
  addBullets(slide, [
    "Recent traffic and weekly recurrence drive most of the predictive accuracy.",
    "XGBoost reached test MAE 158.6 and R² 0.9836.",
    "Congestion classification exceeded 0.94 Macro-F1.",
    "The full workflow reproduces tables, predictions, figures, and tests.",
  ], { left: 80, top: 220, width: 520, height: 300 }, "18pt", C.white);
  addText(slide, "What remains", { left: 690, top: 160, width: 400, height: 44 }, {
    fontSize: "24pt",
    bold: true,
    color: "#F5B88F",
  });
  addBullets(slide, [
    "A long source outage removes much of the early training history.",
    "One traffic counter cannot represent other roads without new validation.",
    "Incident, construction, and event data are absent.",
    "Future work should test multi-step forecasts and uncertainty intervals.",
  ], { left: 690, top: 220, width: 500, height: 300 }, "18pt", C.white);
  addText(slide, "Repository run: 13 tests pass with a single pinned environment.", { left: 220, top: 604, width: 840, height: 44 }, {
    fontSize: "18pt",
    bold: true,
    alignment: "center",
    color: C.white,
  });
  slide.speakerNotes.text = "Sources: results/metrics/regression_test.csv, results/metrics/classification_test.csv, README.md, and NOTES.md.";
}

for (const [index, slide] of presentation.slides.items.entries()) {
  await fs.writeFile(
    path.join(renderDir, `draft-${String(index + 1).padStart(2, "0")}.png`),
    new Uint8Array(await (await presentation.export({ slide, format: "png", scale: 1 })).arrayBuffer()),
  );
}

await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const requirements = {
  explicitTotalSlideCount: 12,
  requiredNativeTableOwnerSlides: [3, 5, 6, 7, 10],
  requiredNativeChartOwnerSlides: [7, 8, 9, 10],
  requiredEmbeddedWorkbookChartOwnerSlides: [],
  materializeLiteralChartWorkbooks: true,
};
const fontPolicy = { basis: "design", families: [FONT] };
await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath: reviewPath,
  pythonExecutable: runtimePython,
  integrityValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
    ...requirements.requiredNativeTableOwnerSlides.flatMap((number) => ["--require-native-table-slide", String(number)]),
  ],
  requiredNativeTableOwnerSlides: requirements.requiredNativeTableOwnerSlides,
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, "traffic_volume_prediction_final_v3.validation.json"),
});

const finalPresentation = await PresentationFile.importPptx(await FileBlob.load(reviewPath));
for (const [index, slide] of finalPresentation.slides.items.entries()) {
  await fs.writeFile(
    path.join(renderDir, `final-${String(index + 1).padStart(2, "0")}.png`),
    new Uint8Array(await (await finalPresentation.export({ slide, format: "png", scale: 1 })).arrayBuffer()),
  );
}

console.log(reviewPath);
