import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const OUT_DIR = path.join(ROOT, "outputs/station_inventory");
const PREVIEW_DIR = path.join(ROOT, "tmp/station_inventory/previews");
const OUTPUT = path.join(OUT_DIR, "金华江联合站点一次性清单与上游关系.xlsx");
const VERBOSE = ["1", "true", "yes", "on"].includes((process.env.PAPERV4_VERBOSE ?? "").toLowerCase());

function reportInspection(label, inspection) {
  if (VERBOSE) {
    console.log(`\n[check] ${label}`);
    console.log(inspection.ndjson);
    return;
  }
  const lines = String(inspection.ndjson ?? "").split("\n").filter(Boolean).length;
  console.log(`[check] ${label} | records=${lines}`);
}

const COLORS = {
  teal: "#0F766E",
  tealDark: "#134E4A",
  tealLight: "#CCFBF1",
  blue: "#1D4ED8",
  blueLight: "#DBEAFE",
  amber: "#B45309",
  amberLight: "#FEF3C7",
  red: "#B91C1C",
  redLight: "#FEE2E2",
  green: "#166534",
  greenLight: "#DCFCE7",
  gray: "#475569",
  grayLight: "#F1F5F9",
  line: "#CBD5E1",
  white: "#FFFFFF",
  ink: "#0F172A",
};

const workbook = Workbook.create();
const overview = workbook.worksheets.add("总览");

const mappings = {
  priority: { must: "必需", recommended: "推荐", conditional: "条件性" },
  category: {
    full_hydrology: "全要素水文",
    special_flow: "专用流量断面",
    rain_only: "连续小时雨量",
    water_level_only: "水位站",
    flow_station: "水文（流量）站",
    reservoir_operation: "水库调度与出库",
  },
  localStatus: {
    no_matching_hourly: "无匹配小时数据",
    daily_discharge_only: "仅有日均流量",
    no_hourly_flow: "无小时流量",
    annual_max_only_not_usable: "仅有年极值且不可用于时序",
    station_name_unknown: "控制站名称待确认",
    not_found_locally: "本地未找到",
    legacy_proxy_mapping: "原25站代理映射",
  },
  sourceGroup: { current: "当前流域补充", legacy: "原25站映射", current_and_legacy: "当前与原映射共有" },
  evidenceStatus: {
    verified: "官方核验",
    verified_new_station: "官方规划新建站",
    not_verified: "未核验",
    rejected: "证据不成立",
    official_station: "官方站表",
    official_station_and_legacy_code: "官方站表+原站码",
    official_modernization_list: "官方改造清单",
    official_new_station: "官方新建站",
    name_and_code_to_verify: "站名站码待确认",
    legacy_proxy_mapping: "原25站代理映射",
    pdf_scan_page_1: "扫描件第1页核验",
  },
  scanAssessment: {
    not_confirmed: "扫描件未确认",
    type_mismatch: "站型不匹配",
    wrong_location_risk: "同名地点风险",
    no_station_name: "非正式站名",
  },
  auditDecision: {
    keep: "保留",
    conditional_keep: "条件保留",
    discovery_only: "仅作数据发现",
    remove_from_request: "从申请清单删除",
  },
  usefulness: { high: "高", medium: "中", low: "低" },
  requestDecision: {
    keep: "保留申请",
    optional: "可选申请",
    do_not_request_now: "本轮不申请",
    do_not_request_for_jinhua: "金华江主图不申请",
  },
  yesNo: { yes: "是", no: "否", true: "是", false: "否" },
  scope: { core: "主图", extension: "汇流扩展", excluded: "排除" },
  status: {
    usable_2022: "2022全年可用",
    usable_2022_partial: "2022年中起可用",
    late_2023: "晚起始（2023年底）",
    missing_hourly: "缺少小时水质",
    excluded_wrong_basin: "排除（非本流域）",
  },
  confidence: { high: "高", medium: "中", low: "低" },
  edgeType: {
    same_reach: "同一连续河段",
    same_reach_reservoir_controlled: "同河段（水库控制）",
    tributary_confluence: "支流汇流",
    direct: "物理直连",
    contracted: "收缩边",
  },
  decision: { replace: "替换", reject_reverse: "拒绝反向" },
  correction: {
    alias: "名称别名",
    county_and_reach: "行政区与河段",
    river: "河流归属",
    reach: "河段归并",
    county: "行政区",
    reach_alias: "河段别名",
    network_status: "控制级别变化",
    county_and_basin: "行政区与流域",
    basin: "流域归属",
  },
};

function excelColumn(index) {
  let n = index + 1;
  let out = "";
  while (n > 0) {
    const r = (n - 1) % 26;
    out = String.fromCharCode(65 + r) + out;
    n = Math.floor((n - 1) / 26);
  }
  return out;
}

function replaceValue(value, map) {
  if (value === null || value === undefined) return value;
  const key = String(value).toLowerCase();
  return Object.prototype.hasOwnProperty.call(map, key) ? map[key] : value;
}

function translateVariables(value) {
  const variableMap = {
    precipitation_mm: "降雨量(mm)",
    water_level_m: "水位(m)",
    discharge_m3_s: "流量(m3/s)",
    velocity_m_s: "平均流速(m/s)",
    cross_section_area_m2: "过水断面面积(m2)",
    rating_curve: "水位流量关系",
    quality_flag: "质量标志",
    inflow_m3_s: "入库流量(m3/s)",
    outflow_m3_s: "出库流量(m3/s)",
    spill_m3_s: "溢洪流量(m3/s)",
    gate_opening: "闸门开度",
  };
  return String(value).split("|").map((item) => variableMap[item] ?? item).join("；");
}

async function importDataSheet(config) {
  const csvText = await fs.readFile(path.join(ROOT, config.csv), "utf8");
  const parsedWorkbook = await Workbook.fromCSV(csvText, { sheetName: config.name });
  const parsedSheet = parsedWorkbook.worksheets.getItem(config.name);
  let values = parsedSheet.getUsedRange().values;
  const sheet = workbook.worksheets.add(config.name);

  values[0] = config.headers;
  if (config.transform) {
    values = values.map((row, rowIndex) => rowIndex === 0 ? row : config.transform(row));
  }
  const rowCount = values.length;
  const colCount = values[0].length;
  const lastCol = excelColumn(colCount - 1);
  const used = sheet.getRange(`A1:${lastCol}${rowCount}`);
  used.values = values;
  const header = sheet.getRange(`A1:${lastCol}1`);
  const body = rowCount > 1 ? sheet.getRange(`A2:${lastCol}${rowCount}`) : null;

  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  header.format = {
    fill: COLORS.tealDark,
    font: { bold: true, color: COLORS.white, size: 10 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: COLORS.tealDark },
  };
  header.format.rowHeight = 34;
  if (body) {
    body.format = {
      font: { color: COLORS.ink, size: 9 },
      verticalAlignment: "top",
      wrapText: true,
      borders: {
        insideHorizontal: { style: "thin", color: COLORS.line },
        bottom: { style: "thin", color: COLORS.line },
      },
    };
    body.format.rowHeight = config.rowHeight ?? 42;
  }

  config.widths.forEach((width, colIndex) => {
    sheet.getRange(`${excelColumn(colIndex)}1:${excelColumn(colIndex)}${rowCount}`).format.columnWidth = width;
  });

  const table = sheet.tables.add(`A1:${lastCol}${rowCount}`, true, config.tableName);
  table.showFilterButton = true;

  if (config.conditionals) config.conditionals(sheet, rowCount);
  return { sheet, rowCount, colCount };
}

const request = await importDataSheet({
  name: "一次申请清单",
  csv: "data/metadata/jinhua_hydromet_request_all.csv",
  tableName: "HydrometRequestTable",
  headers: ["申请编号", "优先级", "数据类别", "站点或断面", "河流或分区", "申请字段", "时间分辨率", "申请时段", "本地现状", "用途", "来源", "备注"],
  widths: [11, 10, 18, 23, 20, 45, 14, 19, 23, 32, 44, 36],
  rowHeight: 54,
  transform: (row) => {
    row[1] = replaceValue(row[1], mappings.priority);
    row[2] = replaceValue(row[2], mappings.category);
    row[5] = translateVariables(row[5]);
    row[6] = String(row[6]).replace("1h_or_finer", "1小时或更细");
    row[7] = String(row[7]).replace("2022-01-01_to_latest", "2022-01-01至最新");
    row[8] = replaceValue(row[8], mappings.localStatus);
    return row;
  },
  conditionals: (sheet, rows) => {
    const range = sheet.getRange(`B2:B${rows}`);
    range.conditionalFormats.add("containsText", { text: "必需", format: { fill: COLORS.redLight, font: { color: COLORS.red, bold: true } } });
    range.conditionalFormats.add("containsText", { text: "推荐", format: { fill: COLORS.amberLight, font: { color: COLORS.amber, bold: true } } });
    range.conditionalFormats.add("containsText", { text: "条件性", format: { fill: COLORS.grayLight, font: { color: COLORS.gray } } });
  },
});

const rainStations = await importDataSheet({
  name: "降雨站完整清单",
  csv: "data/metadata/jinhua_rain_station_request_crosswalk.csv",
  tableName: "RainStationRequestTable",
  headers: ["申请编号", "降雨站", "站码", "来源组", "河流或范围", "关联水质站", "申请时段", "时间分辨率", "优先级", "证据状态", "来源", "备注"],
  widths: [11, 21, 14, 18, 23, 42, 19, 15, 11, 24, 46, 36],
  rowHeight: 52,
  transform: (row) => {
    row[3] = replaceValue(row[3], mappings.sourceGroup);
    row[6] = String(row[6]).replace("2022-01-01_to_latest", "2022-01-01至最新");
    row[7] = String(row[7]).replace("1h_or_finer", "1小时或更细");
    row[8] = replaceValue(row[8], mappings.priority);
    row[9] = replaceValue(row[9], mappings.evidenceStatus);
    return row;
  },
  conditionals: (sheet, rows) => {
    sheet.getRange(`I2:I${rows}`).conditionalFormats.add("containsText", { text: "必需", format: { fill: COLORS.redLight, font: { color: COLORS.red, bold: true } } });
    sheet.getRange(`J2:J${rows}`).conditionalFormats.add("containsText", { text: "待确认", format: { fill: COLORS.amberLight, font: { color: COLORS.amber, bold: true } } });
  },
});

const outsideSources = await importDataSheet({
  name: "扫描件外来源说明",
  csv: "data/metadata/jinhua_hydromet_outside_scan_sources.csv",
  tableName: "OutsideScanSourceTable",
  headers: ["条目编号", "原申请编号", "站点", "原申请数据类型", "扫描件判断", "扫描件观察", "扫描件外来源", "来源位置", "来源地址或路径", "处理", "备注"],
  widths: [11, 12, 24, 21, 19, 43, 34, 22, 50, 25, 48],
  rowHeight: 64,
  transform: (row) => {
    row[4] = replaceValue(row[4], mappings.scanAssessment);
    return row;
  },
  conditionals: (sheet, rows) => {
    const assessment = sheet.getRange(`E2:E${rows}`);
    assessment.conditionalFormats.add("containsText", { text: "未确认", format: { fill: COLORS.redLight, font: { color: COLORS.red, bold: true } } });
    assessment.conditionalFormats.add("containsText", { text: "不匹配", format: { fill: COLORS.amberLight, font: { color: COLORS.amber, bold: true } } });
    assessment.conditionalFormats.add("containsText", { text: "风险", format: { fill: COLORS.amberLight, font: { color: COLORS.amber, bold: true } } });
    assessment.conditionalFormats.add("containsText", { text: "非正式", format: { fill: COLORS.redLight, font: { color: COLORS.red, bold: true } } });
    const action = sheet.getRange(`J2:J${rows}`);
    action.conditionalFormats.add("containsText", { text: "不进入", format: { fill: COLORS.redLight, font: { color: COLORS.red, bold: true } } });
    action.conditionalFormats.add("containsText", { text: "删除", format: { fill: COLORS.redLight, font: { color: COLORS.red, bold: true } } });
    action.conditionalFormats.add("containsText", { text: "仅按", format: { fill: COLORS.amberLight, font: { color: COLORS.amber, bold: true } } });
  },
});

const reservoirUse = await importDataSheet({
  name: "水库研究用途（非申请）",
  csv: "data/metadata/jinhua_reservoir_use_decision.csv",
  tableName: "ReservoirUseDecisionTable",
  headers: ["申请编号", "水库", "物理角色", "影响路径", "当前优先级", "用途强度", "申请决定", "需要字段", "原因"],
  widths: [11, 23, 23, 37, 14, 12, 20, 42, 54],
  rowHeight: 60,
  transform: (row) => {
    row[4] = replaceValue(row[4], mappings.priority);
    row[5] = replaceValue(row[5], mappings.usefulness);
    row[6] = replaceValue(row[6], mappings.requestDecision);
    row[7] = row[7] ? translateVariables(row[7]) : row[7];
    return row;
  },
  conditionals: (sheet, rows) => {
    const decision = sheet.getRange(`G2:G${rows}`);
    decision.conditionalFormats.add("containsText", { text: "保留", format: { fill: COLORS.greenLight, font: { color: COLORS.green, bold: true } } });
    decision.conditionalFormats.add("containsText", { text: "可选", format: { fill: COLORS.amberLight, font: { color: COLORS.amber, bold: true } } });
    decision.conditionalFormats.add("containsText", { text: "不申请", format: { fill: COLORS.redLight, font: { color: COLORS.red, bold: true } } });
  },
});

const nodes = await importDataSheet({
  name: "联合水质节点",
  csv: "data/metadata/jinhua_combined_station_crosswalk.csv",
  tableName: "CombinedWaterQualityNodes",
  headers: ["标准站名", "图中站名", "当前级别", "地图来源", "地图编号", "本地小时数据", "模型数据起点", "城市", "县域或边界", "标准河段", "支系", "图范围", "节点状态", "置信度", "备注"],
  widths: [14, 15, 11, 20, 10, 14, 20, 12, 23, 25, 13, 13, 22, 10, 38],
  rowHeight: 48,
  transform: (row) => {
    row[5] = replaceValue(row[5], mappings.yesNo);
    row[11] = replaceValue(row[11], mappings.scope);
    row[12] = replaceValue(row[12], mappings.status);
    row[13] = replaceValue(row[13], mappings.confidence);
    return row;
  },
  conditionals: (sheet, rows) => {
    const scope = sheet.getRange(`L2:L${rows}`);
    scope.conditionalFormats.add("containsText", { text: "主图", format: { fill: COLORS.greenLight, font: { color: COLORS.green, bold: true } } });
    scope.conditionalFormats.add("containsText", { text: "汇流扩展", format: { fill: COLORS.blueLight, font: { color: COLORS.blue, bold: true } } });
    scope.conditionalFormats.add("containsText", { text: "排除", format: { fill: COLORS.redLight, font: { color: COLORS.red } } });
    const status = sheet.getRange(`M2:M${rows}`);
    status.conditionalFormats.add("containsText", { text: "晚起始", format: { fill: COLORS.amberLight, font: { color: COLORS.amber } } });
    status.conditionalFormats.add("containsText", { text: "缺少", format: { fill: COLORS.redLight, font: { color: COLORS.red, bold: true } } });
  },
});

const physicalEdges = await importDataSheet({
  name: "联合物理边",
  csv: "data/metadata/jinhua_combined_edges_verified.csv",
  tableName: "CombinedPhysicalEdges",
  headers: ["上游站", "下游站", "支系", "边类型", "列出节点中物理相邻", "2023-12后允许建模", "置信度", "证据", "来源", "备注"],
  widths: [14, 14, 20, 19, 19, 19, 10, 35, 45, 42],
  rowHeight: 58,
  transform: (row) => {
    row[3] = replaceValue(row[3], mappings.edgeType);
    row[4] = replaceValue(row[4], mappings.yesNo);
    row[5] = replaceValue(row[5], mappings.yesNo);
    row[6] = replaceValue(row[6], mappings.confidence);
    return row;
  },
  conditionals: (sheet, rows) => {
    const allowed = sheet.getRange(`F2:F${rows}`);
    allowed.conditionalFormats.add("containsText", { text: "是", format: { fill: COLORS.greenLight, font: { color: COLORS.green, bold: true } } });
    allowed.conditionalFormats.add("containsText", { text: "否", format: { fill: COLORS.redLight, font: { color: COLORS.red, bold: true } } });
  },
});

const modelEdges = await importDataSheet({
  name: "2022模型边",
  csv: "data/metadata/jinhua_combined_model_edges_2022.csv",
  tableName: "ModelEdges2022",
  headers: ["上游站", "下游站", "完整物理路径", "边类别", "默认允许", "原因"],
  widths: [15, 15, 38, 16, 13, 58],
  rowHeight: 48,
  transform: (row) => {
    row[3] = replaceValue(row[3], mappings.edgeType);
    row[4] = replaceValue(row[4], mappings.yesNo);
    return row;
  },
  conditionals: (sheet, rows) => {
    const allowed = sheet.getRange(`E2:E${rows}`);
    allowed.conditionalFormats.add("containsText", { text: "是", format: { fill: COLORS.greenLight, font: { color: COLORS.green, bold: true } } });
    allowed.conditionalFormats.add("containsText", { text: "否", format: { fill: COLORS.redLight, font: { color: COLORS.red, bold: true } } });
  },
});

await importDataSheet({
  name: "元数据校正",
  csv: "data/metadata/jinhua_station_overrides.csv",
  tableName: "StationMetadataOverrides",
  headers: ["原站名", "标准站名", "控制级别", "城市", "县域", "标准河流", "标准河段", "图范围", "校正类型", "置信度", "证据", "来源", "备注"],
  widths: [13, 13, 12, 12, 24, 16, 26, 13, 18, 10, 42, 45, 42],
  rowHeight: 60,
  transform: (row) => {
    row[7] = replaceValue(row[7], mappings.scope);
    row[8] = replaceValue(row[8], mappings.correction);
    row[9] = replaceValue(row[9], mappings.confidence);
    return row;
  },
});

await importDataSheet({
  name: "旧边替换",
  csv: "data/metadata/jinhua_combined_old_edge_replacements.csv",
  tableName: "OldEdgeReplacements",
  headers: ["旧上游站", "旧下游站", "替换路径", "处理", "原因"],
  widths: [16, 16, 60, 15, 52],
  rowHeight: 52,
  transform: (row) => {
    row[3] = replaceValue(row[3], mappings.decision);
    return row;
  },
});

overview.showGridLines = false;
overview.getRange("A1:H1").merge();
overview.getRange("A1").values = [["金华江联合站点一次性清单与上游关系"]];
overview.getRange("A1:H1").format = {
  fill: COLORS.tealDark,
  font: { bold: true, color: COLORS.white, size: 18 },
  horizontalAlignment: "left",
  verticalAlignment: "center",
};
overview.getRange("A1:H1").format.rowHeight = 38;

overview.getRange("A2:H2").merge();
overview.getRange("A2").values = [["范围：东阳江、南江、武义江、金华江主图及衢江-兰江汇流扩展；更新时间 2026-07-15"]];
overview.getRange("A2:H2").format = {
  fill: COLORS.tealLight,
  font: { color: COLORS.tealDark, size: 10 },
  verticalAlignment: "center",
};
overview.getRange("A2:H2").format.rowHeight = 26;

overview.getRange("A4:H4").values = [["水质节点", null, "主图节点", null, "汇流扩展", null, "排除节点", null]];
overview.getRange("A5:H5").formulas = [[
  "=COUNTA('联合水质节点'!A2:A200)", null,
  "=COUNTIF('联合水质节点'!L2:L200,\"主图\")", null,
  "=COUNTIF('联合水质节点'!L2:L200,\"汇流扩展\")", null,
  "=COUNTIF('联合水质节点'!L2:L200,\"排除\")", null,
]];
overview.getRange("A4:H5").format.borders = { preset: "outside", style: "thin", color: COLORS.line };
for (const cell of ["A4", "C4", "E4", "G4"]) {
  overview.getRange(cell).format = { fill: COLORS.grayLight, font: { bold: true, color: COLORS.gray }, horizontalAlignment: "center" };
}
for (const cell of ["A5", "C5", "E5", "G5"]) {
  overview.getRange(cell).format = { font: { bold: true, color: COLORS.tealDark, size: 16 }, horizontalAlignment: "center" };
}

overview.getRange("A7:H7").values = [["扫描件内申请项", null, "雨量站", null, "水位站", null, "水文（流量）站", null]];
overview.getRange("A8:H8").formulas = [[
  "=COUNTA('一次申请清单'!A2:A200)", null,
  "=COUNTIF('一次申请清单'!C2:C200,\"连续小时雨量\")", null,
  "=COUNTIF('一次申请清单'!C2:C200,\"水位站\")", null,
  "=COUNTIF('一次申请清单'!C2:C200,\"水文（流量）站\")", null,
]];
overview.getRange("A7:H8").format.borders = { preset: "outside", style: "thin", color: COLORS.line };
for (const cell of ["A7", "C7", "E7", "G7"]) {
  overview.getRange(cell).format = { fill: COLORS.grayLight, font: { bold: true, color: COLORS.gray }, horizontalAlignment: "center" };
}
for (const cell of ["A8", "C8", "E8", "G8"]) {
  overview.getRange(cell).format = { font: { bold: true, color: COLORS.blue, size: 16 }, horizontalAlignment: "center" };
}

overview.getRange("A10:H10").values = [["联合物理边", null, "2022直接允许边", null, "2022默认禁用边", null, "晚起始/缺失节点", null]];
overview.getRange("A11:H11").formulas = [[
  "=COUNTA('联合物理边'!A2:A200)", null,
  "=COUNTIF('2022模型边'!E2:E200,\"是\")", null,
  "=COUNTIF('2022模型边'!E2:E200,\"否\")", null,
  "=COUNTIF('联合水质节点'!M2:M200,\"晚起始（2023年底）\")+COUNTIF('联合水质节点'!M2:M200,\"缺少小时水质\")", null,
]];
overview.getRange("A10:H11").format.borders = { preset: "outside", style: "thin", color: COLORS.line };
for (const cell of ["A10", "C10", "E10", "G10"]) {
  overview.getRange(cell).format = { fill: COLORS.grayLight, font: { bold: true, color: COLORS.gray }, horizontalAlignment: "center", wrapText: true };
}
for (const cell of ["A11", "C11", "E11", "G11"]) {
  overview.getRange(cell).format = { font: { bold: true, color: COLORS.amber, size: 16 }, horizontalAlignment: "center" };
}

overview.getRange("A13:H13").merge();
overview.getRange("A13").values = [["严格上游关系摘要"]];
overview.getRange("A13:H13").format = { fill: COLORS.teal, font: { bold: true, color: COLORS.white, size: 12 } };
const routes = [
  ["东阳江", "梓誉 → 横锦大桥 → 义东桥 → 塔下洲 → 候芹渡 → 低田 → 东关桥"],
  ["南江", "台口 → 三景头 → 岩下 → 方塘 → 画坞坑 → 南江桥 → 候芹渡"],
  ["武义江", "光瑶 → 章店 → 桐琴桥 → 范村 → 焦岩 → 洪坞桥；长安坝 → 范村"],
  ["金华江", "东关桥 + 洪坞桥 → 河盘桥 → 婺城大桥 → 沈村 → 费垅"],
  ["兰江扩展", "下童 → 洋港 → 横山 → 将军岩；费垅 → 将军岩"],
];
routes.forEach((route, i) => {
  const row = 14 + i;
  overview.getRange(`A${row}:B${row}`).merge();
  overview.getRange(`C${row}:H${row}`).merge();
  overview.getRange(`A${row}`).values = [[route[0]]];
  overview.getRange(`C${row}`).values = [[route[1]]];
  overview.getRange(`A${row}:H${row}`).format = {
    fill: i % 2 === 0 ? COLORS.grayLight : COLORS.white,
    font: { color: COLORS.ink, size: 10, bold: i === 0 },
    verticalAlignment: "center",
    wrapText: true,
    borders: { bottom: { style: "thin", color: COLORS.line } },
  };
  overview.getRange(`A${row}:B${row}`).format.font = { bold: true, color: COLORS.tealDark };
  overview.getRange(`A${row}:H${row}`).format.rowHeight = 30;
});

overview.getRange("A20:H20").merge();
overview.getRange("A20").values = [["使用约束"]];
overview.getRange("A20:H20").format = { fill: COLORS.teal, font: { bold: true, color: COLORS.white, size: 12 } };
const rules = [
  "1. 2022 年正式图只使用当时有数据且物理相邻的边；所有跨站收缩边默认禁用。",
  "2. 横锦水库控制边必须同时取得出库流量；不能把梓誉直接跨库传到义东桥。",
  "3. 汇流流量分权只使用当时可观测的小时流量；日均流量不能计算 4 小时传播时延。",
  "4. 正式申请只使用扫描件内可辨认且站型匹配的 30 项：18 个雨量站、2 个水位站、10 个水文（流量）站。",
  "5. 扫描件外或站型不匹配的 34 项只在“扫描件外来源说明”留痕，不进入正式申请；同名不同站型不得替代。",
];
rules.forEach((rule, i) => {
  const row = 21 + i;
  overview.getRange(`A${row}:H${row}`).merge();
  overview.getRange(`A${row}`).values = [[rule]];
  overview.getRange(`A${row}:H${row}`).format = {
    fill: i % 2 === 0 ? COLORS.amberLight : COLORS.white,
    font: { color: COLORS.ink, size: 10 },
    verticalAlignment: "center",
    wrapText: true,
    borders: { bottom: { style: "thin", color: COLORS.line } },
  };
  overview.getRange(`A${row}:H${row}`).format.rowHeight = 30;
});

overview.getRange("A27:H27").merge();
overview.getRange("A27").values = [["颜色说明：红色=必需、禁止或扫描件未确认；黄色=推荐、站型不匹配或需要注意；绿色=可用；蓝色=汇流扩展。状态文字仍是正式判断依据。"]];
overview.getRange("A27:H27").format = { fill: COLORS.grayLight, font: { italic: true, color: COLORS.gray, size: 9 }, wrapText: true };
overview.getRange("A27:H27").format.rowHeight = 28;

overview.freezePanes.freezeRows(2);
for (let col = 0; col < 8; col += 1) {
  overview.getRange(`${excelColumn(col)}1:${excelColumn(col)}27`).format.columnWidth = 16;
}

await fs.mkdir(OUT_DIR, { recursive: true });
await fs.rm(PREVIEW_DIR, { recursive: true, force: true });
await fs.mkdir(PREVIEW_DIR, { recursive: true });

const inspectSummary = await workbook.inspect({
  kind: "table",
  range: "总览!A1:H27",
  include: "values,formulas",
  tableMaxRows: 30,
  tableMaxCols: 10,
  maxChars: 9000,
});
reportInspection("overview", inspectSummary);

const inspectRequest = await workbook.inspect({
  kind: "table",
  range: "一次申请清单!A1:L8",
  include: "values,formulas",
  tableMaxRows: 10,
  tableMaxCols: 12,
  maxChars: 7000,
});
reportInspection("request list", inspectRequest);

const inspectOutsideSources = await workbook.inspect({
  kind: "table",
  range: "扫描件外来源说明!A1:K8",
  include: "values,formulas",
  tableMaxRows: 10,
  tableMaxCols: 11,
  maxChars: 9000,
});
reportInspection("outside sources", inspectOutsideSources);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
reportInspection("formula errors", errors);

const sheetNames = ["总览", "一次申请清单", "降雨站完整清单", "扫描件外来源说明", "水库研究用途（非申请）", "联合水质节点", "联合物理边", "2022模型边", "元数据校正", "旧边替换"];
for (const sheetName of sheetNames) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(path.join(PREVIEW_DIR, `${sheetName}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(OUTPUT);
console.log("\n[completed] station inventory workbook");
console.log(`  result | output=${OUTPUT}`);
console.log(`  rows | requests=${request.rowCount - 1} | rain=${rainStations.rowCount - 1} | outside=${outsideSources.rowCount - 1} | reservoirs=${reservoirUse.rowCount - 1} | nodes=${nodes.rowCount - 1} | physical_edges=${physicalEdges.rowCount - 1} | model_edges=${modelEdges.rowCount - 1}`);
