import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from shapely.geometry import Polygon


st.set_page_config(page_title="Kernel Demo", layout="wide")
st.title("Kernel (nullspace) of a matrix")

st.markdown("""The **kernel** (or **nullspace**) of a matrix $A$ corresponds to the
    set of vectors in the input domain that maps to the **zero vector** in the output domain.""")
cont = st.container(border=True)
with cont:
    st.markdown("""Specifically: for a matrix $A \in \mathbb{R}^{m\\times n}$""")
    st.latex("ker(A) = {\{x\in \mathbb{R}^{n}: A x = 0}\}")

st.markdown("""Trivially, the zero vector is always in the kernel of a matrix (because
    $A0 = 0$).""")

st.markdown("""The **kernel** is important not only because it tells us which vectors
    get 'compressed' to a length of 0 during the transformation with the matrix, but also
    for another reason: it looks different for **regular** and for **singular** matrices.
    How does this work, though?""")

cont = st.container(border=True)
with cont:
    st.markdown("""When mapping a vector with a matrix, we actually take the **scalar 
        product** of rows of the matrix with the vector. Think about it; here, I colored
        the matrix row and vector to visualize that fact: """)
    st.latex(r"""
    \begin{aligned}
    &Ax = \begin{pmatrix}
    \color{red}{a_{11}} & \color{red}{a_{12}} \\
    \color{gray}{a_{21}} & \color{gray}{a_{22}}
    \end{pmatrix}
    \begin{pmatrix}
    \color{red}{x_1} \\
    \color{red}{x_2}
    \end{pmatrix} = 
    \begin{pmatrix}
    \color{red}{a_{11}x_1+a_{12}x_2} \\
    \color{gray}{a_{21}x_1+a_{22}x_2}
    \end{pmatrix} 
    \end{aligned}
    """)

    st.markdown("""
        If we now want to find vectors that are in the **kernel** of our matrix, we have to
        take a look at this equation:
    """)
    st.latex(r"""
        \begin{aligned}
        &Ax = \begin{pmatrix}
        a_{11} & a_{12} \\
        a_{21} & a_{22}
        \end{pmatrix}
        \begin{pmatrix}
        x_1 \\
        x_2
        \end{pmatrix} = 
        \begin{pmatrix}
        0 \\
        0
        \end{pmatrix}
        \end{aligned}
    """)
    st.markdown("""This means, we are looking for a vector that is 0 in the scalar product
        with all rows of $A$ - geometrically speaking, it should be **orthogonal** to the row vectors!""")

col1, col2 = st.columns(2)

with col1:
    cont = st.container(border=True)
    with cont:
        st.badge("In the case of a singular matrix...", color='red')
        st.markdown("""If the matrix $A$ is **singular**, this means there are rows or
            columns that are **linear combinations** of each other - we thus **have not
            found** the full possible set of linearly independent vectors for our space yet!""")
        st.latex(r"""
        \scriptsize
        \textit{
        In $\mathbb{R}^{n \times n}$, we can find up to $n$ vectors that are linearly independent of each other.}""")
        st.latex(r"""
        \scriptsize
        \textit{
        Any additional one will be a linear combination of the ones we already had.}""")

        st.markdown("""As the matrix is singular, we have not found all possible vectors yet;
            we **can** identify another vector that is orthogonal to the others.""")
        st.badge("""The kernel ker(A) is not **trivial** (it contains more than the zero vector)!""", color='red')


with col2:
    cont = st.container(border=True)
    with cont:
        st.badge("In the case of a regular matrix...", color='red')
        st.markdown("""
            If the matrix is **regular**, this means that all rows and columns are 
            **linearly independent** of each other. We thus already **have** a **full set**
            of linearly independent vectors for our space!
        """)
        st.latex(r"""
        \scriptsize
        \textit{
        Any additional vector we find will be \textbf{linearly dependent} on the rows of $A$.}""")
        st.latex(r"""
        \scriptsize
        \textit{
        We will not be able to find another vector that is \textbf{orthogonal} to the rows of $A$ that is not the zero vector.}""")
        st.markdown("""As the matrix is regular, we have already found all possible vectors;
            we cannot find another one that is not the zero vector anymore.""")
        st.badge("""The kernel ker(A) is **trivial** (it contains **exclusively** the zero vector)!""",color='red')