import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# ResidueLab: simple-pole calculator + complex-plane heatmap
# Run with: streamlit run app.py
# ------------------------------------------------------------

st.set_page_config(page_title="ResidueLab", page_icon="∮", layout="wide")

st.title("ResidueLab")
st.caption("Explore rational complex functions, simple poles, residues, and circular contours.")

z = sp.symbols("z")
LOCAL_SYMBOLS = {
    "z": z,
    "I": sp.I,
    "i": sp.I,
    "pi": sp.pi,
    "E": sp.E,
    "sqrt": sp.sqrt,
}

EXAMPLES = {
    "Choose an example…": "1/(z**2 + 1)",
    "1 / (z² + 1)": "1/(z**2 + 1)",
    "1 / ((z - 1)(z + 2))": "1/((z - 1)*(z + 2))",
    "1 / (z(z + 1))": "1/(z*(z + 1))",
    "(z + 1) / (z² + 4)": "(z + 1)/(z**2 + 4)",
    "2 / (z - 1)": "2/(z - 1)",
}

def load_example():
    st.session_state["function_text"] = EXAMPLES[st.session_state["selected_example"]]

if "function_text" not in st.session_state:
    st.session_state["function_text"] = EXAMPLES["Choose an example…"]

with st.sidebar:
    st.header("Function")
    st.selectbox(
        "Load an example function",
        list(EXAMPLES.keys()),
        key="selected_example",
        on_change=load_example,
    )
    function_text = st.text_input(
        "Enter f(z)",
        key="function_text",
        help="Use Python-style powers: z**2. Examples: 1/(z**2 + 1), 1/(z*(z + 1)).",
    )
    st.markdown("**Supported syntax**")
    st.markdown("- Variable: `z`")
    st.markdown("- Powers: `z**2` (or `z^2`)")
    st.markdown("- Operations: `+`, `-`, `*`, `/`, parentheses")
    st.markdown("- Constants: `I` or `i`, `pi`, `E`; function: `sqrt(...)`")
    st.divider()
    st.header("Circular contour")
    radius = st.slider(
        "Radius R",
        min_value=0.25,
        max_value=4.0,
        value=1.5,
        step=0.25,
        help="The contour is the counterclockwise circle |z| = R centred at the origin.",
    )
    plot_limit = st.slider(
        "Heatmap range",
        min_value=2.0,
        max_value=6.0,
        value=3.0,
        step=0.5,
        help="Sets the displayed real and imaginary coordinate range.",
    )
    grid_size = st.select_slider(
        "Heatmap resolution",
        options=[120, 180, 240, 320],
        value=240,
    )

def parse_function(text):
    """Parse a deliberately small, calculator-like expression syntax."""
    cleaned = text.strip().replace("^", "**")
    if not cleaned:
        raise ValueError("Enter a function first.")

    # Permit only calculator-style characters; this is not a general Python evaluator.
    allowed_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        " +-*/().,"
    )
    if any(ch not in allowed_chars for ch in cleaned):
        raise ValueError("Use only letters, numbers, parentheses, spaces, and + - * / **.")

    identifiers = __import__("re").findall(r"[A-Za-z]+", cleaned)
    allowed_names = {"z", "I", "i", "pi", "E", "sqrt"}
    unknown_names = sorted(set(identifiers) - allowed_names)
    if unknown_names:
        raise ValueError("Unsupported name(s): " + ", ".join(unknown_names) + ". Use only z, I, pi, E, and sqrt(...).")

    try:
        expr = sp.sympify(cleaned, locals=LOCAL_SYMBOLS, evaluate=True)
        expr = sp.cancel(sp.together(expr))
    except Exception as exc:
        raise ValueError(f"Could not parse that expression: {exc}") from exc

    if expr.has(sp.Function) and not expr.is_rational_function(z):
        raise ValueError("For this version, use rational functions of z (polynomials divided by polynomials).")
    if not expr.is_rational_function(z):
        raise ValueError("Please enter a rational function of z, such as 1/(z**2 + 1).")

    numerator, denominator = sp.fraction(expr)
    if denominator == 1:
        raise ValueError("This function has no poles. Try a fraction such as 1/(z**2 + 1).")
    if z not in denominator.free_symbols and sp.simplify(denominator).is_number:
        raise ValueError("The denominator must depend on z.")
    return expr, sp.factor(numerator), sp.factor(denominator)

def numeric_complex(value):
    value = complex(sp.N(value, 15))
    if not (np.isfinite(value.real) and np.isfinite(value.imag)):
        raise ValueError("A pole could not be represented numerically.")
    return value

def analyze_poles(expr, denominator):
    """Find denominator roots and residues; reject repeated poles in this starter version."""
    roots = sp.solve(sp.Eq(denominator, 0), z)
    if not roots:
        raise ValueError("Could not find the poles symbolically for this denominator.")

    results = []
    derivative = sp.diff(denominator, z)
    for pole in roots:
        derivative_at_pole = sp.simplify(derivative.subs(z, pole))
        if derivative_at_pole == 0 or derivative_at_pole.equals(0):
            raise ValueError(
                f"The pole at z = {sp.sstr(pole)} is repeated. "
                "This first version supports simple poles only."
            )
        residue = sp.simplify(sp.limit((z - pole) * expr, z, pole))
        results.append({
            "exact_pole": pole,
            "pole": numeric_complex(pole),
            "exact_residue": residue,
            "residue": numeric_complex(residue),
        })
    return results

def evaluate_grid(expr, limit, n):
    x = np.linspace(-limit, limit, n)
    y = np.linspace(-limit, limit, n)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    fn = sp.lambdify(z, expr, modules=["numpy"])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        F = np.asarray(fn(Z), dtype=np.complex128)
        if F.ndim == 0:
            F = np.full(Z.shape, F, dtype=np.complex128)
        magnitude = np.abs(F)
        heat = np.log10(np.maximum(magnitude, 1e-12))
        heat[~np.isfinite(heat)] = np.nan
    return X, Y, heat

def draw_heatmap(expr, poles, radius, limit, n):
    X, Y, heat = evaluate_grid(expr, limit, n)
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(
        X, Y, heat, shading="auto", cmap="magma",
        vmin=-2, vmax=2
    )
    fig.colorbar(mesh, ax=ax, label=r"$\log_{10}|f(z)|$")
    angles = np.linspace(0, 2 * np.pi, 500)
    ax.plot(radius * np.cos(angles), radius * np.sin(angles),
            color="cyan", linewidth=2, label=f"Contour (radius = {radius:g})")
    for idx, item in enumerate(poles):
        p = item["pole"]
        inside = abs(p) < radius - 1e-8
        on = abs(abs(p) - radius) <= 1e-8
        color = "lime" if inside else "white"
        ax.scatter(p.real, p.imag, marker="x", s=100, linewidths=2.5,
                   color=color, zorder=5)
        label = f"Pole {idx+1}"
        ax.annotate(label, (p.real, p.imag), xytext=(6, 6),
                    textcoords="offset points", color="white", fontsize=9)
    ax.axhline(0, color="white", linewidth=0.6, alpha=0.6)
    ax.axvline(0, color="white", linewidth=0.6, alpha=0.6)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.set_title("Magnitude heatmap with poles and circular contour")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig

def contour_integral_numeric(expr, radius, n=4000):
    """Numerically integrate f(z(t))*z'(t) over z=R*exp(i*t), t in [0, 2pi]."""
    fn = sp.lambdify(z, expr, modules=["numpy"])
    t = np.linspace(0, 2 * np.pi, n, endpoint=True)
    points = radius * np.exp(1j * t)
    dz_dt = 1j * radius * np.exp(1j * t)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        vals = np.asarray(fn(points), dtype=np.complex128) * dz_dt
        if vals.ndim == 0:
            vals = np.full(points.shape, vals, dtype=np.complex128)
    if not np.all(np.isfinite(vals)):
        return None
    # Trapezoidal rule for real and imaginary components.
    return np.trapezoid(vals, t)

try:
    expr, numerator, denominator = parse_function(function_text)
    poles = analyze_poles(expr, denominator)
except Exception as error:
    st.error(str(error))
    st.info("Try one of the example functions from the sidebar.")
    st.stop()

st.latex("f(z) = " + sp.latex(expr))

left, right = st.columns([1.5, 1])
with left:
    st.subheader("Complex-plane heatmap")
    st.caption("Colour shows log₁₀|f(z)|. Pole markers are shown as crosses; the cyan circle is the selected contour.")
    fig = draw_heatmap(expr, poles, radius, plot_limit, grid_size)
    st.pyplot(fig, width='stretch')
    plt.close(fig)

with right:
    st.subheader("Poles and residues")
    for idx, item in enumerate(poles, start=1):
        p = item["pole"]
        inside = abs(p) < radius - 1e-8
        on = abs(abs(p) - radius) <= 1e-8
        st.markdown(f"**Pole {idx}**")
        st.latex(f"p_{idx} = {sp.latex(item['exact_pole'])}")
        st.latex(r"\operatorname{Res}(f,p_" + str(idx) + r") = " + sp.latex(item["exact_residue"]))
        st.caption(f"Numerical pole: {p.real:.4g} {'+' if p.imag >= 0 else '−'} {abs(p.imag):.4g}i")
        if on:
            st.warning("This pole lies on (or extremely close to) the contour. The standard Residue Theorem cannot be applied to this contour.")
        elif inside:
            st.success("Inside the contour")
        else:
            st.info("Outside the contour")
        st.divider()

    enclosed = [
        item for item in poles
        if abs(item["pole"]) < radius - 1e-8
    ]
    on_contour = [
        item for item in poles
        if abs(abs(item["pole"]) - radius) <= 1e-8
    ]

    st.subheader("Contour integral")
    st.caption("The circle is traversed counterclockwise and centred at the origin.")
    if on_contour:
        st.error("Cannot apply the standard theorem: a pole lies on the selected contour. Change the radius.")
    else:
        residue_sum = sp.simplify(sum((item["exact_residue"] for item in enclosed), sp.S.Zero))
        exact_integral = sp.simplify(2 * sp.pi * sp.I * residue_sum)
        st.markdown("**Sum of enclosed residues**")
        st.latex(r"\sum \operatorname{Res}(f,p_k) = " + sp.latex(residue_sum))
        st.markdown("**Residue Theorem result**")
        st.latex(r"\oint_C f(z)\,dz = 2\pi i\sum \operatorname{Res}(f,p_k)")
        st.latex(r"\boxed{\oint_C f(z)\,dz = " + sp.latex(exact_integral) + "}")

        numerical = contour_integral_numeric(expr, radius)
        if numerical is None:
            st.warning("Numerical integration could not be evaluated; the function may be singular on the contour.")
        else:
            st.markdown("**Numerical contour-integration check**")
            st.code(f"{numerical.real:.8g} {'+' if numerical.imag >= 0 else '-'} {abs(numerical.imag):.8g}i")
            exact_num = complex(sp.N(exact_integral, 15))
            error = abs(numerical - exact_num)
            st.caption(f"Absolute difference: {error:.3e}")

st.divider()
st.subheader("How to interpret the result")
st.markdown(
    """
    1. The heatmap displays the magnitude of the function over the complex plane.
    2. The crosses mark poles. A pole contributes to the contour integral only when it is inside the selected circle.
    3. Add the residues of all enclosed poles, then multiply the sum by `2πi`.
    4. The numerical check directly approximates the integral around the circle and should agree when the contour does not pass through a pole.
    """
)
st.info(
    "Scope: this learning version is designed for rational functions with simple poles. "
    "It does not support repeated poles, and the contour must not pass through a pole."
)
