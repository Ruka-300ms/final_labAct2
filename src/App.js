// InventoryDashboard.jsx
import React, { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";

/*Helper: Data Augmentation */
const createAugmentedProducts = (rawProducts = []) =>
  rawProducts.map((p) => {
    const baseInv = Math.round(p.price * 2.5 + p.id * 5);
    const avgSales = Math.round(baseInv * (0.1 + Math.random() * 0.2)); // 10-30%
    const leadTime = Math.round(1 + Math.random() * 5); // 1..6 days

    return {
      id: p.id,
      name: p.title.length > 40 ? p.title.substring(0, 40) + "..." : p.title,
      currentInventory: baseInv,
      averageSalesPerWeek: avgSales,
      daysToReplenish: leadTime,
      reorderPrediction: "Pending",
      mlFeatures: [baseInv, avgSales, leadTime],
    };
  });

/* Model training helper*/
const buildAndTrainModel = async (onEpochEnd = null) => {
  // Training data 
  const trainingFeatures = tf.tensor2d([
    [20, 50, 3],
    [5, 30, 5],
    [15, 40, 4],
    [8, 60, 2],
  ]);
  const trainingLabels = tf.tensor2d([[0], [1], [0], [1]]);

  // Model architecture
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [3], units: 8, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  model.compile({
    optimizer: "adam",
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  // Train
  await model.fit(trainingFeatures, trainingLabels, {
    epochs: 200,
    shuffle: true,
    callbacks: onEpochEnd ? { onEpochEnd } : undefined,
  });

  // Dispose training tensors to avoid leaks
  trainingFeatures.dispose();
  trainingLabels.dispose();

  return model;
};

//Small presentational components for the dashboard layout.
const HeaderBar = ({ title }) => (
  <header style={styles.header}>
    <div style={styles.headerInner}>
      <h1 style={{ margin: 0, fontSize: 20 }}>{title}</h1>
      <div style={styles.headerSubtitle}>Quick ML-based reorder suggestions</div>
    </div>
  </header>
);

const StatCard = ({ label, value, accent }) => (
  <div style={{ ...styles.card, borderLeft: `6px solid ${accent}` }}>
    <div style={styles.cardLabel}>{label}</div>
    <div style={styles.cardValue}>{value}</div>
  </div>
);

const Toolbar = ({ onTrain, onPredict, isTrained }) => (
  <div style={styles.toolbar}>
    <button
      onClick={onTrain}
      disabled={isTrained}
      style={{
        ...styles.button,
        background: isTrained ? "#9bbf9b" : "#0b6f9b",
        cursor: isTrained ? "default" : "pointer",
      }}
    >
      {isTrained ? "Model Trained" : "1 — Train Model"}
    </button>
    <button
      onClick={onPredict}
      disabled={!isTrained}
      style={{
        ...styles.button,
        background: isTrained ? "#36454f" : "#9aa0a6",
        cursor: isTrained ? "pointer" : "default",
      }}
    >
      2 — Run Predictions
    </button>
    <div style={{ marginLeft: 12, alignSelf: "center" }}>
      <span style={{ fontWeight: 600 }}>Status:</span>{" "}
      <span style={{ color: isTrained ? "#087f07" : "#b00020" }}>
        {isTrained ? "Ready for Prediction" : "Model needs training"}
      </span>
    </div>
  </div>
);


   //Main Component

export default function InventoryDashboard() {
  const [items, setItems] = useState([]); 
  const [loading, setLoading] = useState(true);
  const [isModelReady, setModelReady] = useState(false);
  const [tfModel, setTfModel] = useState(null);
  const [totalReorder, setTotalReorder] = useState(0); 

  // Fetch & augment products on mount
  useEffect(() => {
    const loadProducts = async () => {
      try {
        const res = await fetch("https://fakestoreapi.com/products?limit=100");
        const json = await res.json();
        const augmented = createAugmentedProducts(json);
        setItems(augmented);
      } catch (err) {
        console.error("Failed loading products:", err);
      } finally {
        setLoading(false);
      }
    };
    loadProducts();
  }, []);

  // Train model handler: builds and trains the model then saves it
  const handleTrainModel = async () => {
    setModelReady(false);
    try {
      const model = await buildAndTrainModel();
      setTfModel(model);
      setModelReady(true);
      alert("Model trained successfully! Ready to predict.");
    } catch (err) {
      console.error("Training error:", err);
      alert("Error training model (check console).");
    }
  };

  // Run predictions across items, using the same threshold (=0.5)
  const handlePredict = async () => {
    if (!tfModel) {
      alert("Please train the model first!");
      return;
    }

    let reorderCount = 0;
    const updated = items.map((it) => {
      // features -> tensor (shape [1,3])
      const features = tf.tensor2d([it.mlFeatures]);

      // model.predict returns a tensor; we read value synchronously
      const res = tfModel.predict(features);

      // read the scalar probability
      const prob = res.dataSync()[0];

      // cleanup tensors
      features.dispose();
      res.dispose();

      const suggestion = prob >= 0.5 ? "REORDER" : "No Reorder";
      if (suggestion === "REORDER") reorderCount++;

      return {
        ...it,
        reorderPrediction: suggestion,
      };
    });

    setItems(updated);
    setTotalReorder(reorderCount);
  };

  if (loading) {
    return (
      <div style={styles.loaderWrap}>
        <div style={styles.loaderBox}>Loading products from API...</div>
      </div>
    );
  }

  return (
    <div style={styles.page}>
      <HeaderBar title="Inventory Reorder Predictor" />

      <main style={styles.main}>
        <section style={styles.topRow}>
          <div style={styles.leftColumn}>
            <Toolbar
              onTrain={handleTrainModel}
              onPredict={handlePredict}
              isTrained={isModelReady}
            />

            <div style={styles.cardRow}>
              <StatCard
                label="Products (sample)"
                value={items.length}
                accent="#0b6f9b"
              />
              <StatCard
                label="Items flagged REORDER"
                value={totalReorder}
                accent="#b22b2b"
              />
              <StatCard
                label="Model status"
                value={isModelReady ? "Trained" : "Not trained"}
                accent={isModelReady ? "#087f07" : "#b00020"}
              />
            </div>
          </div>

          <div style={styles.rightColumn}>
            <div style={styles.infoBox}>
              <div style={{ fontWeight: 700 }}>Goal</div>
              <div style={{ marginTop: 6, fontSize: 13 }}>
                Predict whether a product needs reordering (REORDER) based on:
                Inventory, Average Weekly Sales, and Lead Time. (Binary
                classifier, threshold 0.5)
              </div>
            </div>
          </div>
        </section>

        <section style={styles.tableSection}>
          <table style={styles.table}>
            <thead>
              <tr style={styles.tableHeadRow}>
                <th style={styles.th}>Product Name</th>
                <th style={styles.th}>Current Inventory</th>
                <th style={styles.th}>Avg. Weekly Sales</th>
                <th style={styles.th}>Lead Time (days)</th>
                <th style={styles.th}>Reorder</th>
              </tr>
            </thead>
            <tbody>
              {items.map((row) => (
                <tr key={row.id} style={styles.tr}>
                  <td style={styles.td}>{row.name}</td>
                  <td style={styles.td}>{row.currentInventory}</td>
                  <td style={styles.td}>{row.averageSalesPerWeek}</td>
                  <td style={styles.td}>{row.daysToReplenish}</td>
                  <td
                    style={{
                      ...styles.td,
                      fontWeight: 700,
                      color: row.reorderPrediction === "REORDER" ? "#b22b2b" : "#1f7a24",
                    }}
                  >
                    {row.reorderPrediction}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </main>
    </div>
  );
}


   //Local styles (self-contained)
const styles = {
  page: {
    fontFamily: "Inter, Arial, sans-serif",
    color: "#222",
    background: "#f5f7f8",
    minHeight: "100vh",
  },
  header: {
    background: "linear-gradient(90deg,#083b57,#0b6f9b)",
    color: "#fff",
    padding: "14px 20px",
    boxShadow: "0 2px 8px rgba(2,8,23,0.08)",
  },
  headerInner: {
    maxWidth: 1100,
    margin: "0 auto",
  },
  headerSubtitle: {
    fontSize: 12,
    opacity: 0.9,
    marginTop: 4,
  },

  main: {
    maxWidth: 1100,
    margin: "20px auto",
    padding: "0 18px 60px",
  },

  topRow: {
    display: "flex",
    gap: 20,
    marginBottom: 18,
  },

  leftColumn: {
    flex: 1,
    minWidth: 0,
  },

  rightColumn: {
    width: 320,
  },

  toolbar: {
    display: "flex",
    alignItems: "center",
    gap: 12,
    marginBottom: 12,
  },

  button: {
    padding: "10px 14px",
    borderRadius: 8,
    color: "#fff",
    border: "none",
    fontWeight: 600,
  },

  cardRow: {
    display: "flex",
    gap: 12,
    marginTop: 12,
  },

  card: {
    flex: 1,
    padding: 12,
    background: "#fff",
    borderRadius: 8,
    boxShadow: "0 1px 6px rgba(12,21,30,0.04)",
  },

  cardLabel: { fontSize: 12, color: "#586069" },
  cardValue: { marginTop: 6, fontSize: 18 },

  infoBox: {
    background: "#fff",
    padding: 12,
    borderRadius: 8,
    boxShadow: "0 1px 6px rgba(12,21,30,0.04)",
  },

  tableSection: {
    marginTop: 18,
    background: "#fff",
    borderRadius: 8,
    boxShadow: "0 1px 10px rgba(12,21,30,0.04)",
    overflowX: "auto",
  },

  table: {
    width: "100%",
    borderCollapse: "collapse",
    minWidth: 800,
  },

  tableHeadRow: {
    background: "#f4f7f9",
  },

  th: {
    textAlign: "left",
    padding: "12px 10px",
    fontSize: 13,
    borderBottom: "1px solid #eef1f4",
  },

  tr: {
    borderBottom: "1px solid #f1f4f6",
  },

  td: {
    padding: "10px 10px",
    fontSize: 14,
  },

  loaderWrap: {
    minHeight: "200px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },

  loaderBox: {
    padding: 20,
    background: "#fff",
    borderRadius: 8,
    boxShadow: "0 1px 8px rgba(0,0,0,0.06)",
  },
};
