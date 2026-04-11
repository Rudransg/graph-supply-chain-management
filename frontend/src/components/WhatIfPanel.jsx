import { useState, useEffect, useRef, useCallback } from "react";
import ForceGraph3D from "react-force-graph-3d";
import { runWhatIf, fetchFactoryLoad,fetchGraphEdges } from "../api";
import SpriteText from "three-spritetext";  // npm install three-spritetext



// ── color system ──────────────────────────────────────────────────────────────
const NODE_COLORS = {
  product:  "#f59e0b",   // amber  — products
  plant:    "#22d3ee",   // cyan   — plants (factories)
  storage:  "#a78bfa",   // purple — storage locations
};
const NODE_BORDER = {
  product:  "#d97706",
  plant:    "#0891b2",
  storage:  "#e60f13",
};
const SIGNAL_COLORS = {
  production_unit:  "#38bdf8",
  delivery_unit:    "#34d399",
  sales_order_unit: "#f59e0b",
};
const SIGNAL_LABELS = {
  production_unit:  "Production",
  delivery_unit:    "Delivery",
  sales_order_unit: "Sales Order",
};

// ── build graph data from products + factories ────────────────────────────────
function buildGraphData(products, factories, edges = []) {
  const nodes = [];
  const links = [];
  const productSet = new Set(products);

  const plantLinks = new Set();
  const storageLinks = new Set();
  const storageSet = new Set();

  factories.forEach((fac) => {
    nodes.push({ id: `plant_${fac}`, label: fac, type: "plant", val: 15 });
  });

  edges.forEach((edge) => {
    if (edge.type === "same_storage" && edge.storage) {
      storageSet.add(edge.storage);
    }
  });

  [...storageSet].forEach((storage) => {
    nodes.push({
      id: `storage_${storage}`,
      label: storage,
      type: "storage",
      val: 11
    });
  });

  products.forEach((prod) => {
    nodes.push({ id: prod, label: prod, type: "product", val: 4 });
  });

  edges.forEach((edge) => {
    if (!productSet.has(edge.source) || !productSet.has(edge.target)) return;

    if (
      edge.type === "same_product_group" ||
      edge.type === "same_product_subgroup"
    ) {
      links.push({
        source: edge.source,
        target: edge.target,
        type: edge.type
      });
    }

    if (edge.type === "same_plant" && edge.plant) {
      const plantId = `plant_${edge.plant}`;
      const keyA = `${edge.source}__${plantId}`;
      const keyB = `${edge.target}__${plantId}`;

      if (!plantLinks.has(keyA)) {
        plantLinks.add(keyA);
        links.push({ source: edge.source, target: plantId, type: "product_plant" });
      }
      if (!plantLinks.has(keyB)) {
        plantLinks.add(keyB);
        links.push({ source: edge.target, target: plantId, type: "product_plant" });
      }
    }

    if (edge.type === "same_storage" && edge.storage) {
      const storageId = `storage_${edge.storage}`;
      const keyA = `${edge.source}__${storageId}`;
      const keyB = `${edge.target}__${storageId}`;

      if (!storageLinks.has(keyA)) {
        storageLinks.add(keyA);
        links.push({ source: edge.source, target: storageId, type: "product_storage" });
      }
      if (!storageLinks.has(keyB)) {
        storageLinks.add(keyB);
        links.push({ source: edge.target, target: storageId, type: "product_storage" });
      }
    }
  });

  return { nodes, links };
}

export default function WhatIfPanel({ products = [], factories = [] }) {
  const fgRef = useRef();
  const [graphData, setGraphData]     = useState({ nodes: [], links: [] });
  const [selectedProduct, setProduct] = useState("");
  const [zeroProd,   setZeroProd]     = useState(new Set());
  const [zeroFact,   setZeroFact]     = useState(new Set());
  const [result,     setResult]       = useState(null);
  const [loading,    setLoading]       = useState(false);
  const [error,      setError]         = useState("");
  const [hoveredNode, setHovered]      = useState(null);
  const [camDistance, setCamDistance] = useState(400);
  const [edges, setEdges] = useState([]);
  const [productSearch, setProductSearch] = useState("");
    const [plantSearch, setPlantSearch] = useState("");

    const allProducts = graphData.nodes
    .filter((n) => n.type === "product")
    .map((n) => n.id)
    .sort();

    const allPlants = graphData.nodes
    .filter((n) => n.type === "plant")
    .map((n) => n.label || n.id)
    .sort();

    const filteredProducts = allProducts.filter((p) =>
    p.toLowerCase().includes(productSearch.toLowerCase())
    );

    const filteredPlants = allPlants.filter((p) =>
    p.toLowerCase().includes(plantSearch.toLowerCase())
    );
    const allFilteredProductsSelected =
  filteredProducts.length > 0 &&
  filteredProducts.every((p) => zeroProd.has(p));

const allFilteredPlantsSelected =
  filteredPlants.length > 0 &&
  filteredPlants.every((p) => zeroFact.has(p));

const toggleAllFilteredProducts = () => {
  setZeroProd((prev) => {
    const next = new Set(prev);
    if (allFilteredProductsSelected) {
      filteredProducts.forEach((p) => next.delete(p));
    } else {
      filteredProducts.forEach((p) => next.add(p));
    }
    return next;
  });
};

const toggleAllFilteredPlants = () => {
  setZeroFact((prev) => {
    const next = new Set(prev);
    if (allFilteredPlantsSelected) {
      filteredPlants.forEach((p) => next.delete(p));
    } else {
      filteredPlants.forEach((p) => next.add(p));
    }
    return next;
  });
};
  const [visibleRelationTypes, setVisibleRelationTypes] = useState(
    new Set([
        "same_product_group",
        "same_product_subgroup",
        "product_plant",
        "product_storage",
    ])
    );
  const lastUpdateRef = useRef(0);

  // build graph once products/factories arrive
  useEffect(() => {
  if (products.length && factories.length && edges.length) {
    const fullData = buildGraphData(products, factories, edges);

    const filteredLinks = fullData.links.filter(link =>
      visibleRelationTypes.has(link.type)
    );

    setGraphData({ nodes: fullData.nodes, links: filteredLinks });

    if (!selectedProduct && products.length) {
      setProduct(products[0]);
    }
  }
}, [products, factories, edges, visibleRelationTypes, selectedProduct]);
  useEffect(() => {
  fetchGraphEdges()
    .then((res) => {
      console.log("Edges from API:", res.data.edges.length, res.data.edges[0]);
      setEdges(res.data.edges);
    })
    .catch((err) => console.error("fetchGraphEdges failed:", err));
}, []);

  useEffect(() => {
    if (fgRef.current) {
      fgRef.current.d3Force('charge').strength(-550);
      fgRef.current.d3Force('link').distance(2000);
    }
  }, [graphData]);
  // node click — toggle zeroed state
  const handleNodeClick = useCallback((node) => {
    if (node.type === "product") {
      setZeroProd((prev) => {
        const next = new Set(prev);
        next.has(node.id) ? next.delete(node.id) : next.add(node.id);
        return next;
      });
    }
    if (node.type === "plant") {
      const facName = node.label;
      setZeroFact((prev) => {
        const next = new Set(prev);
        next.has(facName) ? next.delete(facName) : next.add(facName);
        return next;
      });
    }
  }, []);

  // draw nodes
  const paintNode = useCallback(
    (node, ctx, globalScale) => {
      const isZeroedProd = node.type === "product" && zeroProd.has(node.id);
      const isZeroedFact = node.type === "plant"   && zeroFact.has(node.label);
      const isSelected   = node.id === selectedProduct;
      const isHovered    = hoveredNode?.id === node.id;

      const r = node.type === "plant" ? 10 : node.type === "storage" ? 7 : 6;

      // glow for selected/hovered
      if (isSelected || isHovered) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, r + 5, 0, 2 * Math.PI);
        ctx.fillStyle = isSelected ? "rgba(56,189,248,0.2)" : "rgba(255,255,255,0.1)";
        ctx.fill();
      }

      // node shape
      ctx.beginPath();
      if (node.type === "plant") {
        // square for plants
        ctx.rect(node.x - r, node.y - r, r * 2, r * 2);
      } else if (node.type === "storage") {
        // triangle for storage
        ctx.moveTo(node.x, node.y - r);
        ctx.lineTo(node.x + r, node.y + r);
        ctx.lineTo(node.x - r, node.y + r);
        ctx.closePath();
      } else {
        // circle for products
        ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
      }

      // fill — dim if zeroed
      ctx.fillStyle = isZeroedProd || isZeroedFact
        ? "rgba(127,29,29,0.8)"   // red tint when zeroed
        : NODE_COLORS[node.type] || "#fbfbfb";
      ctx.fill();

      // border
      ctx.strokeStyle = isZeroedProd || isZeroedFact
        ? "#ef4444"
        : isSelected ? "#38bdf8"
        : NODE_BORDER[node.type] || "#64748b";
      ctx.lineWidth = isSelected ? 2.5 : 1.5;
      ctx.stroke();

      // label
      const fontSize = Math.max(3, 10 / globalScale);
      ctx.font = `${fontSize}px sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = isZeroedProd || isZeroedFact ? "#fca5a5" : "#e2e8f0";
      ctx.fillText(node.label, node.x, node.y + r + fontSize + 1);
    },
    [zeroProd, zeroFact, selectedProduct, hoveredNode]
  );

  async function runScenario() {
    if (!selectedProduct) { setError("Select a product first"); return; }
    setLoading(true); setError(""); setResult(null);
    try {
      const res = await runWhatIf({
        product_name:      selectedProduct,
        zeroed_products:   [...zeroProd],
        zeroed_factories:  [...zeroFact],
        capacity_overrides: {},
        dropped_relations: [],
      });
      console.log("whatif response:", res.data);
      setResult(res.data);
    } catch (e) {
      setError(e.response?.data?.detail ?? e.message);
    } finally {
      setLoading(false);
    }
  }

  const zeroedCount = zeroProd.size + zeroFact.size;

const focusNodeIds = new Set([
  ...(selectedProduct ? [selectedProduct] : []),
  ...[...zeroProd].map((p) => String(p)),
  ...[...zeroFact].map((f) => `plant_${String(f)}`),
]);

const highlightedNodeIds = new Set();
const highlightedLinkKeys = new Set();

graphData.links.forEach((link) => {
  const sourceId =
    typeof link.source === "object" ? link.source.id : link.source;
  const targetId =
    typeof link.target === "object" ? link.target.id : link.target;

  if (focusNodeIds.has(sourceId) || focusNodeIds.has(targetId)) {
    highlightedNodeIds.add(sourceId);
    highlightedNodeIds.add(targetId);
    highlightedLinkKeys.add(`${sourceId}__${targetId}__${link.type}`);
    highlightedLinkKeys.add(`${targetId}__${sourceId}__${link.type}`);
  }
});
  const selectedRoot = selectedProduct ? String(selectedProduct).trim() : null;
const flowLevel1 = [];
const flowLevel2 = [];

if (selectedRoot) {
  const level1Ids = new Set();
  const level2Ids = new Set();

  const allowedFlowTypes = new Set([
    "product_storage",
    "same_product_group",
    "same_product_subgroup",
  ]);

  const getKey = (v) => {
    if (!v) return "";
    if (typeof v === "object") return String(v.id ?? v.label ?? "").trim();
    return String(v).trim();
  };

  const nodeMap = new Map();
  graphData.nodes.forEach((n) => {
    const idKey = getKey(n.id);
    const labelKey = getKey(n.label);
    if (idKey) nodeMap.set(idKey, n);
    if (labelKey) nodeMap.set(labelKey, n);
  });

  const validLinks = graphData.links.filter((link) =>
    allowedFlowTypes.has(link.type)
  );

  // 1‑hop neighbors
  validLinks.forEach((link) => {
    const sourceKey = getKey(link.source);
    const targetKey = getKey(link.target);

    if (sourceKey === selectedRoot) level1Ids.add(targetKey);
    if (targetKey === selectedRoot) level1Ids.add(sourceKey);
  });

  // 2‑hop neighbors
  validLinks.forEach((link) => {
    const sourceKey = getKey(link.source);
    const targetKey = getKey(link.target);

    if (
      level1Ids.has(sourceKey) &&
      targetKey !== selectedRoot &&
      !level1Ids.has(targetKey)
    ) {
      level2Ids.add(targetKey);
    }

    if (
      level1Ids.has(targetKey) &&
      sourceKey !== selectedRoot &&
      !level1Ids.has(sourceKey)
    ) {
      level2Ids.add(sourceKey);
    }
  });

  [...level1Ids].forEach((key) => {
    const node = nodeMap.get(String(key));
    if (node) flowLevel1.push(node);
  });

  [...level2Ids].forEach((key) => {
    const node = nodeMap.get(String(key));
    if (node) flowLevel2.push(node);
  });
}
  console.log([...new Set(graphData.links.map((l) => l.type))]);
  console.log("selectedRoot:", selectedRoot);
  console.log("flowLevel1:", flowLevel1.map(n => n.id || n.label));
  console.log("flowLevel2:", flowLevel2.map(n => n.id || n.label));
  return (
    <div style={S.wrapper}>
      {/* ── LEFT: graph ── */}
      <div style={S.graphPanel}>
        <div style={S.graphHeader}>
          <p style={S.graphTitle}>Graph — Click nodes to toggle offline</p>
          <div style={S.legend}>
            <LegendItem color={NODE_COLORS.product}  shape="circle"  label="Product" />
            <LegendItem color={NODE_COLORS.plant}    shape="square"  label="Plant" />
            <LegendItem color={NODE_COLORS.storage}  shape="triangle" label="Storage" />
            <LegendItem color="#ef4444" shape="circle" label="Zeroed / Offline" />
          </div>
        </div>

        <ForceGraph3D
            ref={fgRef}
            graphData={graphData}
             d3AlphaDecay={0.02}
            d3VelocityDecay={0.3}
            d3Force="charge"
            onEngineStop={() => fgRef.current?.d3Force('charge').strength(-120)}
            onNodeClick={handleNodeClick}
            onNodeHover={setHovered}
            nodeThreeObject={(node) => {
                const sprite = new SpriteText(node.label);
                const isZeroed =
                (node.type === "product" && zeroProd.has(node.id)) ||
                (node.type === "plant"   && zeroFact.has(node.label));
                sprite.color           = isZeroed ? "#fca5a5" : NODE_COLORS[node.type] || "#94a3b8";
                sprite.textHeight = Math.max(2, Math.min(20, camDistance / 35));
                sprite.backgroundColor = "rgba(7,13,26,0.6)";
                sprite.padding         = 2;
                sprite.borderRadius    = 3;
                return sprite;
            }}
            nodeThreeObjectExtend={true}
            onCameraPositionChange={(pos) => {
                const now = Date.now();
                if (now - lastUpdateRef.current > 100) {
                    lastUpdateRef.current = now;
                    setCamDistance(Math.sqrt(pos.x ** 2 + pos.y ** 2 + pos.z ** 2));
                }
            }}
            nodeColor={(node) => {
            const isForecast       = node.id === selectedProduct;
            const isZeroedProduct  = node.type === "product" && zeroProd.has(node.id);
            const isZeroedPlant    = node.type === "plant"   && zeroFact.has(node.label);
            const isHighlighted    = highlightedNodeIds.has(node.id);

            if (isZeroedProduct || isZeroedPlant) return "#ef4444";   // disrupted
            if (isForecast) return "#ffffff";                         // forecast-for product
            if (isHighlighted) return NODE_COLORS[node.type] || "#94a3b8";
            if (node.type === "product") return "rgba(245,158,11,0.35)";
            if (node.type === "plant")   return "rgba(34,211,238,0.35)";
            if (node.type === "storage") return "rgba(167,139,250,0.35)";
            return "#94a3b8";
            }}
            nodeVal={(node) => node.type === "plant" ? 8 : 4}
            linkColor={(link) => {
                const sourceId    = typeof link.source === "object" ? link.source.id : link.source;
                const targetId    = typeof link.target === "object" ? link.target.id : link.target;
                const isHighlighted = highlightedLinkKeys.has(`${sourceId}__${targetId}__${link.type}`);
                if (isHighlighted) {
                    return link.type === "same_product_group"      ? "#f59e0b" :        // amber
                            link.type === "same_product_subgroup"   ? "#8b5cf6" :        // violet
                            link.type === "product_plant"           ? "#22d3ee" :
                            link.type === "product_storage"         ? "#e569ad" :
                            "#e2e8f0";
                    }

                return link.type === "same_product_group"        ? "rgba(245,158,11,0.25)" :
                    link.type === "same_product_subgroup"     ? "rgba(139,92,246,0.25)" :
                    link.type === "product_plant"             ? "rgba(34,211,238,0.10)" :
                    link.type === "product_storage"           ? "rgba(167,139,250,0.10)" :
                    "rgba(148,163,184,0.08)";
                }}

            linkWidth={(link) => {
                const sourceId    = typeof link.source === "object" ? link.source.id : link.source;
                const targetId    = typeof link.target === "object" ? link.target.id : link.target;
                const isHighlighted = highlightedLinkKeys.has(`${sourceId}__${targetId}__${link.type}`);

                return isHighlighted ? 3.5 :
                        link.type === "product_storage" ? 1.2 :
                        link.type === "product_plant"   ? 1.1 :
                        0.8;
                }}                       // was 1 — thicker lines
            linkOpacity={1}                     // was 0.5 — fully opaque
            linkDirectionalArrowLength={6}      // adds arrowheads so direction is clear
            linkDirectionalArrowRelPos={1}      // arrow at the target end
            linkDirectionalParticles={(link) => {
                const sourceId    = typeof link.source === "object" ? link.source.id : link.source;
                const targetId    = typeof link.target === "object" ? link.target.id : link.target;
                const isHighlighted = highlightedLinkKeys.has(`${sourceId}__${targetId}__${link.type}`);

                return isHighlighted ? 3 : 0;   // particles only on highlighted relations
                }}        // animated dots flowing along edges ✨
            linkDirectionalParticleSpeed={0.005}
            linkDirectionalParticleWidth={2}
            backgroundColor="#0c111f"
            width={520}
            height={480}
        />
        <div style={S.flowPanel}>
  <p style={S.flowTitle}>Connection flow</p>

  {!selectedProduct ? (
    <p style={S.flowEmpty}>
      Select a product to see its animated 2-hop connection flow.
    </p>
  ) : (
    <div style={S.flowLane}>
      <div style={S.flowColumn}>
        <p style={S.flowLabel}>Selected</p>
        <div style={{ ...S.flowNode, ...S.flowNodeRoot, opacity: 1 }}>
          {selectedProduct}
        </div>
      </div>

      <div style={S.flowArrowWrap}>
        <div style={S.flowArrowLine} />
        <div style={S.flowArrowDot} />
      </div>

      <div style={S.flowColumn}>
        <p style={S.flowLabel}>Direct</p>
        <div style={S.flowNodeStack}>
          {flowLevel1.length === 0 ? (
            <span style={S.flowMuted}>No direct neighbors</span>
          ) : (
            flowLevel1.slice(0, 8).map((node, idx) => (
            <div
                key={`l1-${node.id || node.label || idx}`}
                style={{
                ...S.flowNode,
                ...S.flowNodeLevel1,
                opacity: 1,
                animationDelay: `${idx * 80}ms`,
                }}
            >
                {node.label || node.id}
                <span style={S.flowNodeType}>{node.type}</span>
            </div>
            ))
          )}
        </div>
      </div>

      <div style={S.flowArrowWrap}>
        <div style={S.flowArrowLine} />
        <div style={S.flowArrowDot} />
      </div>

      <div style={S.flowColumn}>
        <p style={S.flowLabel}>2-hop</p>
        <div style={S.flowNodeStack}>
          {flowLevel2.length === 0 ? (
            <span style={S.flowMuted}>No second-degree neighbors</span>
          ) : (
            flowLevel2.slice(0, 10).map((node, idx) => (
            <div
                key={`l2-${node.id || node.label || idx}`}
                style={{
                ...S.flowNode,
                ...S.flowNodeLevel2,
                opacity: 1,
                animationDelay: `${idx * 80}ms`,
                }}
            >
                {node.label || node.id}
                <span style={S.flowNodeType}>{node.type}</span>
            </div>
            ))
          )}
        </div>
      </div>
    </div>
  )}
</div>
      </div>

      {/* ── RIGHT: controls + results ── */}
      <div style={S.controlPanel}>
  <h2 style={S.heading}>What-If Scenario Builder</h2>
  <p style={S.sub}>Click nodes in the graph to simulate disruptions.</p>

  <div style={S.field}>
    <p style={S.label}>Relations shown</p>
    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
      {[
        ["same_product_group", "Product group"],
        ["same_product_subgroup", "Product subgroup"],
        ["product_plant", "Plant"],
        ["product_storage", "Storage"],
      ].map(([value, label]) => {
        const checked = visibleRelationTypes.has(value);
        return (
          <label
            key={value}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              padding: "4px 10px",
              borderRadius: 999,
              border: checked ? "1px solid #3b82f6" : "1px solid #1e293b",
              background: checked ? "rgba(37,99,235,0.18)" : "#020617",
              fontSize: 11,
              color: "#e2e8f0",
              cursor: "pointer",
            }}
          >
            <input
              type="checkbox"
              checked={checked}
              onChange={() =>
                setVisibleRelationTypes((prev) => {
                  const next = new Set(prev);
                  if (next.has(value)) next.delete(value);
                  else next.add(value);
                  return next;
                })
              }
            />
            {label}
          </label>
        );
      })}
    </div>
  </div>
  <div style={S.field}>
  <p style={S.label}>Manual disruption picker</p>

  <div style={{ display: "grid", gap: 14 }}>
    {/* Products */}
    <div>
      <p style={{ ...S.label, fontSize: 12, marginBottom: 6 }}>Products</p>
      <input
        type="text"
        placeholder="Search products..."
        value={productSearch}
        onChange={(e) => setProductSearch(e.target.value)}
        style={S.select}
      />

     <div style={S.pickList}>
  <label style={{ ...S.pickItem, fontWeight: 700, borderBottom: "1px solid #1e293b", paddingBottom: 8 }}>
    <input
      type="checkbox"
      checked={allFilteredProductsSelected}
      onChange={toggleAllFilteredProducts}
    />
    <span>All visible products</span>
  </label>

  {filteredProducts.map((p) => {
          const checked = zeroProd.has(p);
          return (
            <label key={p} style={S.pickItem}>
              <input
                type="checkbox"
                checked={checked}
                onChange={() =>
                  setZeroProd((prev) => {
                    const next = new Set(prev);
                    if (next.has(p)) next.delete(p);
                    else next.add(p);
                    return next;
                  })
                }
              />
              <span>{p}</span>
            </label>
          );
        })}
      </div>
    </div>

    {/* Plants */}
    <div>
      <p style={{ ...S.label, fontSize: 12, marginBottom: 6 }}>Plants</p>
      <input
        type="text"
        placeholder="Search plants..."
        value={plantSearch}
        onChange={(e) => setPlantSearch(e.target.value)}
        style={S.select}
      />

    <div style={S.pickList}>
  <label style={{ ...S.pickItem, fontWeight: 700, borderBottom: "1px solid #1e293b", paddingBottom: 8 }}>
    <input
      type="checkbox"
      checked={allFilteredPlantsSelected}
      onChange={toggleAllFilteredPlants}
    />
    <span>All visible plants</span>
  </label>

  {filteredPlants.map((f) => {
          const checked = zeroFact.has(f);
          return (
            <label key={f} style={S.pickItem}>
              <input
                type="checkbox"
                checked={checked}
                onChange={() =>
                  setZeroFact((prev) => {
                    const next = new Set(prev);
                    if (next.has(f)) next.delete(f);
                    else next.add(f);
                    return next;
                  })
                }
              />
              <span>{f}</span>
            </label>
          );
        })}
      </div>
    </div>
  </div>
</div>
  {/* forecast for */}
  <div style={S.field}>
          <p style={S.label}>Forecast For</p>
          <select
            value={selectedProduct}
            onChange={(e) => setProduct(e.target.value)}
            style={S.select}
          >
            {products.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>

        {/* active scenario summary */}
        <div style={S.scenarioSummary}>
          <p style={S.label}>Active Disruptions</p>
          {zeroedCount === 0 ? (
            <p style={S.none}>None — click graph nodes to add disruptions</p>
          ) : (
            <div style={S.tagRow}>
              {[...zeroProd].map((p) => (
                <span key={p} style={S.tagRed}>
                  📦 {p}
                  <button style={S.tagX} onClick={() =>
                    setZeroProd(prev => { const n = new Set(prev); n.delete(p); return n; })
                  }>×</button>
                </span>
              ))}
              {[...zeroFact].map((f) => (
                <span key={f} style={S.tagRed}>
                  🏭 {f}
                  <button style={S.tagX} onClick={() =>
                    setZeroFact(prev => { const n = new Set(prev); n.delete(f); return n; })
                  }>×</button>
                </span>
              ))}
            </div>
          )}
          {zeroedCount > 0 && (
            <button style={S.clearBtn} onClick={() => { setZeroProd(new Set()); setZeroFact(new Set()); }}>
              Clear all
            </button>
          )}
        </div>

        <button onClick={runScenario} disabled={loading} style={S.runBtn}>
          {loading ? "Running…" : "▶ Run Scenario"}
        </button>

        {error && <p style={S.errorText}>{error}</p>}

        {/* results */}
        {result && (
          <div style={S.results}>
            <p style={S.label}>Impact on {result.product_name}</p>
            {Object.keys(result.baseline).filter((signal) => signal !== "sales_order").map((signal) => {
              const base  = result.baseline[signal];
              const scen  = result.scenario[signal];
              const delta = result.delta[signal];
              const pct   = result.delta_pct[signal];
              const drop  = delta < 0;
              const barPct = Math.min(100, Math.abs(pct));

              return (
                <div key={signal} style={S.resultCard}>
                  <div style={S.resultTop}>
                    <span style={{ ...S.signalDot, background: SIGNAL_COLORS[signal] }} />
                    <span style={S.signalName}>{SIGNAL_LABELS[signal]}</span>
                    <span style={{ ...S.deltaBadge,
                      background: drop ? "#2d1a1a" : "#1a2d1a",
                      color:      drop ? "#f87171" : "#4ade80",
                      border:     `1px solid ${drop ? "#7f1d1d" : "#14532d"}`,
                    }}>
                      {drop ? "▼" : "▲"} {Math.abs(pct)}%
                    </span>
                  </div>

                  <div style={S.compareNums}>
                    <div>
                      <p style={S.numLabel}>Baseline</p>
                      <p style={{ ...S.numVal, color: SIGNAL_COLORS[signal] }}>{base.toFixed(0)}</p>
                    </div>
                    <span style={S.arrow}>→</span>
                    <div>
                      <p style={S.numLabel}>Scenario</p>
                      <p style={{ ...S.numVal, color: drop ? "#f87171" : "#4ade80" }}>{scen.toFixed(0)}</p>
                    </div>
                  </div>

                  {/* impact bar */}
                  <div style={S.impactTrack}>
                    <div style={{
                      ...S.impactFill,
                      width: `${barPct}%`,
                      background: drop ? "#ef4444" : "#22c55e",
                    }} />
                  </div>

                  {Math.abs(pct) > 40 && (
                    <p style={S.warning}>⚠ Large disruption — may exceed training distribution</p>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// ── legend helper ─────────────────────────────────────────────────────────────
function LegendItem({ color, shape, label }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <svg width="14" height="14" viewBox="0 0 14 14">
        {shape === "circle"   && <circle cx="7" cy="7" r="5" fill={color} />}
        {shape === "square"   && <rect x="2" y="2" width="10" height="10" fill={color} />}
        {shape === "triangle" && <polygon points="7,1 13,13 1,13" fill={color} />}
      </svg>
      <span style={{ color: "#64748b", fontSize: 11 }}>{label}</span>
    </div>
  );
}

// ── styles ────────────────────────────────────────────────────────────────────
const S = {
  flowPanel: {
  marginTop: 14,
  padding: 16,
  border: "1px solid #1e293b",
  borderRadius: 18,
  background: "linear-gradient(180deg, #081225 0%, #0b1730 100%)",
  boxShadow: "0 10px 30px rgba(0,0,0,0.25)",
  overflow: "visible",
},

flowTitle: {
  color: "#f8fafc",
  fontSize: 15,
  fontWeight: 700,
  marginBottom: 14,
},

flowLane: {
  display: "grid",
  gridTemplateColumns: "1fr 60px 1.3fr 60px 1.3fr",
  alignItems: "start",
  gap: 10,
  minHeight: "auto",
},

flowColumn: {
  display: "flex",
  flexDirection: "column",
  gap: 10,
},

flowLabel: {
  color: "#94a3b8",
  fontSize: 12,
  marginBottom: 4,
},

flowNodeStack: {
  display: "flex",
  flexDirection: "column",
  gap: 8,
  minWidth: 0,
},

flowNode: {
  padding: "10px 12px",
  borderRadius: 14,
  fontSize: 12,
  color: "#e2e8f0",
  border: "1px solid rgba(148,163,184,0.18)",
  animation: "fadeSlideIn 480ms ease forwards",
  opacity: 1,
  wordBreak: "break-word",
},
flowNodeRoot: {
  background: "linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)",
  border: "1px solid #3b82f6",
  color: "#eff6ff",
  fontWeight: 700,
},

flowNodeLevel1: {
  background: "rgba(34,211,238,0.12)",
  border: "1px solid rgba(34,211,238,0.24)",
},

flowNodeLevel2: {
  background: "rgba(167,139,250,0.12)",
  border: "1px solid rgba(167,139,250,0.24)",
},

flowNodeType: {
  marginLeft: 8,
  color: "#94a3b8",
  fontSize: 11,
},

flowArrowWrap: {
  position: "relative",
  height: 60,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  marginTop: 28,
},

flowArrowLine: {
  width: 42,
  height: 2,
  borderRadius: 999,
  background: "linear-gradient(90deg, rgba(59,130,246,0.15), rgba(34,211,238,0.9))",
  boxShadow: "0 0 12px rgba(34,211,238,0.35)",
},

flowArrowDot: {
  position: "absolute",
  width: 8,
  height: 8,
  borderRadius: "50%",
  background: "#67e8f9",
  boxShadow: "0 0 10px rgba(103,232,249,0.8)",
  animation: "flowPulse 1.8s infinite ease-in-out",
},

flowMuted: {
  color: "#64748b",
  fontSize: 12,
},

flowEmpty: {
  color: "#64748b",
  fontSize: 12,
  margin: 0,
},

  pickList: {
  marginTop: 8,
  maxHeight: 140,
  overflowY: "auto",
  border: "1px solid #1e293b",
  borderRadius: 12,
  background: "#020617",
  padding: 8,
  display: "grid",
  gap: 6,
 },

  pickItem: {
  display: "flex",
  alignItems: "center",
  gap: 8,
  padding: "6px 8px",
  borderRadius: 8,
  color: "#e2e8f0",
  fontSize: 12,
 },
  wrapper:     { display: "flex", gap: 24, alignItems: "flex-start", flexWrap: "wrap" },
  graphPanel:  { background: "#070d1a", border: "1px solid #1e293b", borderRadius: 12, overflow: "hidden", flex: "0 0 520px" },
  graphHeader: { padding: "14px 18px", borderBottom: "1px solid #1e293b" },
  graphTitle:  { color: "#94a3b8", fontSize: 12, marginBottom: 10 },
  legend:      { display: "flex", gap: 16, flexWrap: "wrap" },
  controlPanel:{ flex: 1, minWidth: 300 },
  heading:     { color: "#e2e8f0", fontSize: 20, fontWeight: 700, marginBottom: 6 },
  sub:         { color: "#475569", fontSize: 13, marginBottom: 20 },
  field:       { marginBottom: 18 },
  label:       { color: "#64748b", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 8 },
  select:      { background: "#0d1424", color: "#e2e8f0", border: "1px solid #334155", borderRadius: 8, padding: "10px 14px", fontSize: 13, minWidth: 200 },
  scenarioSummary: { background: "#0d1424", border: "1px solid #1e293b", borderRadius: 10, padding: "14px 16px", marginBottom: 18 },
  none:        { color: "#334155", fontSize: 12, fontStyle: "italic" },
  tagRow:      { display: "flex", flexWrap: "wrap", gap: 8 },
  tagRed:      { background: "#2d1a1a", border: "1px solid #7f1d1d", color: "#fca5a5", borderRadius: 999, padding: "4px 10px", fontSize: 12, display: "flex", alignItems: "center", gap: 6 },
  tagX:        { background: "none", border: "none", color: "#f87171", cursor: "pointer", fontSize: 14, padding: 0 },
  clearBtn:    { marginTop: 10, background: "none", border: "1px solid #334155", color: "#64748b", borderRadius: 6, padding: "4px 12px", fontSize: 11, cursor: "pointer" },
  runBtn:      { background: "#1d4ed8", color: "#fff", border: "none", borderRadius: 8, padding: "12px 28px", fontSize: 14, fontWeight: 600, cursor: "pointer", width: "100%", marginBottom: 16 },
  errorText:   { color: "#f87171", fontSize: 13, marginBottom: 12 },
  results:     { display: "flex", flexDirection: "column", gap: 12 },
  resultCard:  { background: "#0d1424", border: "1px solid #1e293b", borderRadius: 10, padding: "14px 16px" },
  resultTop:   { display: "flex", alignItems: "center", gap: 8, marginBottom: 10 },
  signalDot:   { width: 8, height: 8, borderRadius: "50%", flexShrink: 0 },
  signalName:  { color: "#94a3b8", fontSize: 12, flex: 1 },
  deltaBadge:  { borderRadius: 999, padding: "2px 10px", fontSize: 11, fontWeight: 600 },
  compareNums: { display: "flex", alignItems: "center", gap: 16, marginBottom: 10 },
  numLabel:    { color: "#475569", fontSize: 10, marginBottom: 2 },
  numVal:      { fontSize: 22, fontWeight: 700 },
  arrow:       { color: "#334155", fontSize: 16 },
  impactTrack: { height: 4, background: "#1e293b", borderRadius: 999, overflow: "hidden" },
  impactFill:  { height: "100%", borderRadius: 999, transition: "width 0.4s ease" },
  warning:     { color: "#f59e0b", fontSize: 11, marginTop: 8 },
};