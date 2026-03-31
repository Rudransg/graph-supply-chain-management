const STAGES = [
  {
    name:   "Data Ingestion",
    file:   "src/data_ingestion.py",
    detail: "Downloads nodes, edges, temporal CSVs from GCS bucket: supply-graph-stuff",
    status: "done",
  },
  {
    name:   "Data Preprocessing",
    file:   "src/datapreprocessing.py",
    detail: "Rolling mean (window=30), builds HeteroData graph, saves X.npy / hetero_data.pt",
    status: "done",
  },
  {
    name:   "Model Training",
    file:   "src/model_training.py",
    detail: "Trains SupplyGraphModel (SAGEConv × 2, hidden=32) with asymmetric loss α=2.0, 50 epochs",
    status: "done",
  },
  {
    name:   "MLflow Logging",
    file:   "mlruns/",
    detail: "Logs MAE, MSE, RMSE, R², asymmetric loss + saves hetero_sage_model.pt",
    status: "done",
  },
  {
    name:   "Docker Build",
    file:   "backend/Dockerfile + frontend/Dockerfile",
    detail: "Backend (python:3.11-slim) + Frontend (node:20-alpine → nginx:alpine)",
    status: "done",
  },
  {
    name:   "Jenkins CI/CD",
    file:   "Jenkinsfile",
    detail: "Build → Test → Push to GCR → SSH deploy to Google Cloud VM",
    status: "running",
  },
  {
    name:   "Google Cloud Deploy",
    file:   "docker-compose.yml",
    detail: "Frontend :3000 + Backend :8000 on GCP VM via docker-compose",
    status: "pending",
  },
];

const COLOR  = { done: "#4ade80", running: "#fbbf24", pending: "#334155" };
const ICON   = { done: "✓",       running: "⟳",       pending: "○"       };

export default function PipelinePanel() {
  return (
    <div>
      <h2 style={S.heading}>CI/CD Pipeline</h2>
      <p style={S.sub}>
        End-to-end pipeline: GCS ingestion → preprocessing → HeteroSAGE training →
        MLflow tracking → Docker → Jenkins → Google Cloud.
      </p>

      <div style={S.list}>
        {STAGES.map((stage, i) => (
          <div key={i} style={S.item}>
            {/* connector line */}
            {i < STAGES.length - 1 && <div style={S.connector} />}

            <div
              style={{
                ...S.iconCircle,
                background:   COLOR[stage.status] + "22",
                border:       `2px solid ${COLOR[stage.status]}`,
                color:        COLOR[stage.status],
              }}
            >
              {ICON[stage.status]}
            </div>

            <div style={S.content}>
              <div style={S.stageRow}>
                <span style={S.stageName}>{stage.name}</span>
                <span
                  style={{
                    ...S.badge,
                    background: COLOR[stage.status] + "22",
                    color:      COLOR[stage.status],
                  }}
                >
                  {stage.status}
                </span>
              </div>
              <p style={S.file}>{stage.file}</p>
              <p style={S.detail}>{stage.detail}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

const S = {
  heading:   { color: "#e2e8f0", fontSize: 22, fontWeight: 700, marginBottom: 8 },
  sub:       { color: "#64748b", fontSize: 13, marginBottom: 28 },
  list:      { display: "flex", flexDirection: "column", gap: 0 },
  item: {
    display: "flex", alignItems: "flex-start",
    gap: 16, position: "relative",
    paddingBottom: 24,
  },
  connector: {
    position: "absolute", left: 19,
    top: 40, bottom: 0,
    width: 2, background: "#1e293b",
    zIndex: 0,
  },
  iconCircle: {
    width: 40, height: 40, borderRadius: "50%",
    display: "flex", alignItems: "center",
    justifyContent: "center", fontSize: 15,
    fontWeight: 700, flexShrink: 0, zIndex: 1,
  },
  content:   { flex: 1 },
  stageRow:  { display: "flex", alignItems: "center", gap: 10, marginBottom: 2 },
  stageName: { color: "#e2e8f0", fontSize: 14, fontWeight: 600 },
  badge: {
    fontSize: 10, fontWeight: 600,
    padding: "2px 8px", borderRadius: 20,
    textTransform: "uppercase", letterSpacing: "0.05em",
  },
  file:   { color: "#475569", fontSize: 11, fontFamily: "monospace", marginBottom: 3 },
  detail: { color: "#64748b", fontSize: 12 },
};
