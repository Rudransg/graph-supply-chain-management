# Graph for Supply Chain Demand Forecasting

This repository implements a heterogeneous graph-based temporal model for supply chain demand forecasting using PyTorch Geometric and PyTorch Geometric Temporal.[page:1]

The code builds a product-level supply graph with multiple relation types (plant, storage location, product group, product subgroup), rolls production time series, and trains a heterogeneous GNN for one-step-ahead forecasting with an asymmetric loss that penalizes under-prediction more than over-prediction.[page:1]

## Use Cases

This project can be applied in a variety of real-world supply chain and demand planning scenarios:

- **Demand Forecasting**: Predict future production quantities for individual products by leveraging both structural (graph) and temporal (time series) patterns in the supply network.
- **Supply Chain Risk Analysis**: Identify products or nodes in the supply graph that are critical bottlenecks by analyzing graph connectivity across plant, storage location, product group, and subgroup relations.
- **Inventory Optimization**: Use one-step-ahead forecasts with an asymmetric loss to reduce the risk of stock-outs, prioritizing avoidance of under-prediction in safety-stock calculations.
- **Network-Aware Planning**: Incorporate relational context (e.g., shared plant or product group) into demand estimates, capturing ripple effects that traditional time-series models miss.
- **Benchmarking GNN Architectures**: Serve as a testbed for evaluating heterogeneous GNN variants (e.g., `HeteroGCN`, `HGT`) on a real-world industrial supply chain dataset.
- **Academic Research**: Supports experimentation with temporal graph learning, heterogeneous graph construction, and asymmetric loss functions in the context of operations research.
## Features

- Construction of a heterogeneous **supply** graph over products using CSV edge lists for:
  - Same plant
  - Same storage location
  - Same product group
  - Same product subgroup[page:1]
- Integration of temporal production data as node-level time series, using rolling-window smoothing over raw production values.[page:1]
- Conversion of the heterogeneous graph to a homogeneous edge index for use with `StaticGraphTemporalSignal` from `torch_geometric_temporal`.[page:1]
- Heterogeneous GNN model (`HeteroGCN`) using `HeteroConv` with `SAGEConv` layers over the four relation types.[page:1]
- Asymmetric loss function that penalizes under-forecasting more strongly via a tunable parameter `alpha`.[page:1]
- Train/test temporal split with basic evaluation (MSE, MAE, R², RMSE) on the forecasting task.[page:1]
- 
## Sources

This project builds on prior work in graph neural networks, temporal graph learning, and supply chain analytics.

### Primary Dataset & Paper (CIOL Research Lab)

- **SupplyGraph: A Benchmark Dataset for Supply Chain Planning using Graph Neural Networks**
  Azmine Toushik Wasi, MD Shafikul Islam, and Adipto Raihan Akib (2024).
  - arXiv: https://arxiv.org/abs/2401.15299
  - GitHub: https://github.com/ciol-researchlab/SupplyGraph
  - BibTeX:
    ```bibtex
    @misc{wasi2024supplygraph,
      title={SupplyGraph: A Benchmark Dataset for Supply Chain Planning using Graph Neural Networks},
      author={Azmine Toushik Wasi and MD Shafikul Islam and Adipto Raihan Akib},
      year={2024},
      eprint={2401.15299},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
    }
    ```

- **Graph Neural Networks in Supply Chain Analytics and Optimization: Concepts, Perspectives, Dataset and Benchmarks**
  Azmine Toushik Wasi, MD Shafikul Islam, Adipto Raihan Akib, and Mahathir Mohammad Bappy (2024).
  Shahjalal University of Science and Technology & Louisiana State University, Computational Intelligence and Operations Laboratory (CIOL).
  - arXiv: https://arxiv.org/abs/2411.08550
  - DOI: https://doi.org/10.48550/arXiv.2411.08550
  - GitHub: https://github.com/ciol-researchlab/SCG
  - BibTeX:
    ```bibtex
    @misc{wasi2024graphneuralnetworkssupply,
      title={Graph Neural Networks in Supply Chain Analytics and Optimization: Concepts, Perspectives, Dataset and Benchmarks},
      author={Azmine Toushik Wasi and MD Shafikul Islam and Adipto Raihan Akib and Mahathir Mohammad Bappy},
      year={2024},
      eprint={2411.08550},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.08550}
    }
    ```

### GNN Backbone References

- Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR.
  https://arxiv.org/abs/1609.02907

- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs (GraphSAGE).* NeurIPS.
  https://arxiv.org/abs/1706.02216

- Schlichtkrull, M., et al. (2018). *Modeling Relational Data with Graph Convolutional Networks (R-GCN / HeteroConv).* ESWC.
  https://arxiv.org/abs/1703.06103

- Wu, Z., et al. (2020). *Graph WaveNet for Deep Spatial-Temporal Graph Modeling.* IJCAI.
  https://arxiv.org/abs/1906.00121

- Rozemberczki, B., et al. (2021). *PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine Learning Models.* CIKM.
  https://arxiv.org/abs/2104.07788

### Libraries & Tools

- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- PyTorch Geometric Temporal: https://pytorch-geometric-temporal.readthedocs.io/
- scikit-learn: https://scikit-learn.org/

## Data Layout

The code assumes a local Google Drive–style structure:

- `Raw Dataset/Homogenoeus/Edges/EdgesIndex/`
  - `Edges (Plant).csv`
  - `Edges (Storage Location).csv`
  - `Edges (Product Group).csv`
  - `Edges (Product Sub-Group).csv`[page:1]
- `Raw Dataset/Homogenoeus/Nodes/`
  - `NodesIndex.csv`[page:1]
- `Raw Dataset/Homogenoeus/Temporal Data/Weight/`
  - `Production .csv` (with a `Date` column that is dropped before rolling)[page:1]
- `Raw Dataset/Homogenoeus/Temporal Data/Unit/`
  - `Production .csv` (used for building the temporal signal)[page:1]
- `Code/Developing heterogeneous graph/`
  - `data.csv`[page:1]

Update these paths if you are not using Colab/Drive.

## Installation

```bash
pip install torch-geometric-temporal
# Also install: torch, torch-geometric, numpy, pandas, matplotlib, scikit-learn
```[page:1]

The script currently installs `torch-geometric-temporal` inline with:

```python
!pip install torch-geometric-temporal
```[page:1]

Adapt this to your environment (e.g., local virtualenv).

## File Overview

### `supply_graph.py`

1. **Data loading and preprocessing**
   - Loads edge lists for each relation type and constructs a `HeteroData` graph object.[page:1]
   - Loads weight-based production data, applies a 30-day rolling mean, drops the first 30 rows, and resets the index.[page:1]
   - Builds a mapping from product columns to indices and ensures alignment with `NodesIndex.csv`.[page:1]

2. **Graph construction**
   - Defines product node features as random vectors `data['rolled_prod'].x` with dimension 16.[page:1]
   - Adds four relation types:
     - `('rolled_prod', 'same_plant', 'product')`
     - `('rolled_prod', 'same_storage', 'product')`
     - `('rolled_prod', 'same_product_group', 'product')`
     - `('rolled_prod', 'same_product_subgroup', 'product')`[page:1]

3. **Temporal data preparation**
   - Loads unit-based `Production .csv` from `Temporal Data/Unit`, separates and drops the `Date` column, and reuses the rolled production values as temporal features.[page:1]
   - Shapes values to `features[t] = production at time t` and `targets[t] = production at time t+1`, each of size `[num_nodes, 1]`.[page:1]

4. **Temporal graph dataset**
   - Concatenates all heterogeneous edge indices into a single homogeneous `edge_index`.[page:1]
   - Wraps the data into a `StaticGraphTemporalSignal` dataset and splits it into train/test with `temporal_signal_split`.[page:1]

5. **Model: `HeteroGCN`**
   - Uses `HeteroConv` with `SAGEConv` for each of the four relation types, aggregating by sum.[page:1]
   - Normalizes edge types so that destination nodes are consistently `'rolled_prod'` rather than `'product'` for compatibility with the feature dictionary.[page:1]
   - Two GNN layers are followed by a linear layer mapping to a scalar per node.[page:1]

6. **Loss and training**
   - Defines an asymmetric loss:
     - Under-prediction (`y_hat < y_true`) is penalized with factor `alpha > 1`.
     - Over-prediction uses a standard squared error.[page:1]
   - Trains the model for 50 epochs by concatenating static node features with the current temporal scalar and minimizing the asymmetric loss.[page:1]
   - Evaluates on the test set and then computes MSE, MAE, R², and RMSE using `sklearn.metrics`.[page:1]

## Usage

1. Place all CSVs in the expected directories or modify the hardcoded paths in `supply_graph.py`.[page:1]
2. Ensure that the product columns in the production CSV match the order in `NodesIndex.csv`.[page:1]
3. Run:

```bash
python supply_graph.py
```[page:1]

In Colab, mount Drive, update `ROOT` and other paths if needed, and then run all cells.

## Asymmetric Loss

The asymmetric loss emphasizes the cost of under-forecasting, which is often more critical in supply chain settings.

Formally, with predictions \\(y_{\\hat{}}\\) and ground truth \\(y\\):

\\[
\\text{loss} = \\text{mean}(\\alpha (y_{\\hat{}} - y)^2 \\mathbf{1}_{y_{\\hat{}} < y} + (y_{\\hat{}} - y)^2 \\mathbf{1}_{y_{\\hat{}} \\ge y})
\\]

where \\(\\alpha > 1\\) scales the penalty when forecasts are below actuals.[page:1]

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric
- PyTorch Geometric Temporal
- NumPy
- pandas
- matplotlib
- scikit-learn[page:1]
```
# Credits for the project
Created by Rudransh Raizada and Yajur Tandon
