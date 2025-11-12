import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.cm as cm
import matplotlib.colors as mcolors

st.set_page_config(page_title="Pulse models", layout="wide")
st.title("Pulse models")

st.markdown(r"""If we want to understand the **temporal dynamics** of a system, we 
    can investigate how the model responds to a **disturbance** or **pulse**. This is a 
    vector input that is transformed by the effect matrix $A$ that models the system.  
    We **simulate** how a disturbance propagates through the system, identifying the
    **long-term** response of the system to the disturbance:""")
st.latex(r"""\vec{p}_k = A\vec{p}_{k-1} = A^k\vec{p}_0""")
st.markdown(r"""As stated, the long-term stability of the system (modelled by the matrix $A$)
    **does not** depend on the initial disturbance (the vector $\vec{p}_0$) but is an
    **intrinsic** property of the effect matrix $A$.""")

cont = st.container(border=True)
with cont:
    st.badge("Pulse stability and convergence", color='red')
    st.markdown(r"""The system is called **pulse stable** if the **spectral radius** 
        (the **largest eigenvalue** in the absolute value) $\rho(A) \leq 1$.  
        This means that the system might never be able to 'get rid' of the disturbance
        (it might remain in the system forever), but the response will not escalate and 
        keep growing indefinitely, either.""")
    st.markdown(r"""The system is **not pulse stable** if the **spectral radius** 
        (the **largest eigenvalue** in the absolute value) $\rho(A) > 1$.  
        This means that a disturbance will grow over time. The system responds in an
        escalating fashion to the disturbance, possibly leading to system destruction
        (think about the military marching across a bridge at the wrong step frequency:
        in the worst case, the bridge might collapse).""")
    st.markdown(r"""The system is called **convergent** if the **spectral radius** 
        (the **largest eigenvalue** in the absolute value) $\rho(A) < 1$.  
        This means that the system will be able to **absord** the disturbance completely,
        meaning the disturbance converges to $0$ over time.""")
    

st.markdown(r"""Apart from the long-term response of a system to a disturbance via the
    spectral radius, we can also investigate the **short-term** dynamics.""")
cont = st.container(border=True)
with cont:
    st.badge("'Contracting' and 'non-contracting' systems'", color='red')
    st.markdown(r"""A system modelled by effect matrix $A$ is called **contracting** if
        the **spectral norm** (the largest singular value of $A$) $||A||_S = \sigma_{max}(A)<1$.  
        This means that, **in each time step**, a pulse (disturbance) is **compressed**
        (i.e., reduced in size). This is **stronger** than pulse stability, where we only
        ask if there is convergence as $t\rightarrow\infty$!""")