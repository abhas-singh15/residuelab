# ResidueLab

An interactive Python application for exploring the Residue Theorem using Streamlit.

## Features

- Enter rational complex functions.
- Calculate simple poles and their residues symbolically.
- Visualize function magnitude using a complex-plane heatmap.
- Select a circular contour by adjusting its radius.
- Calculate contour integrals using the Residue Theorem.
- Compare analytical results with numerical contour integration.

## Tech Stack

- Python
- Streamlit
- SymPy
- NumPy
- Matplotlib

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd residuelab
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Limitations

The current version supports rational functions with simple poles. The circular contour is centred at the origin, and poles must not lie on the contour.

## Mathematical Foundation

For a suitable closed contour C,

\[
\oint_C f(z)\,dz = 2\pi i \sum_k \operatorname{Res}(f,p_k)
\]

where the sum includes the poles enclosed by the contour.

## Engineering Motivation

Residue calculations are useful in evaluating inverse Laplace transforms and analysing electrical circuits and control systems. An RC circuit demonstration can be added as an extension.