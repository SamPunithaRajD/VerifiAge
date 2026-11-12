import streamlit as st

st.set_page_config(page_title="VerifiAge", page_icon="◆", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        background-color: #0B0B0B; border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
        border-radius: 8px; margin: 1px 0;
    }
    html body [data-testid="stSidebarNavLink"][aria-current="page"] {
        background-color: rgba(57,255,20,0.12) !important;
        border-left: 2px solid #39FF14 !important;
    }
    html body [data-testid="stSidebarNavLink"][aria-current="page"] p {
        color: #39FF14 !important;
    }
</style>
<div style="padding: 0.6rem 0 1.2rem 0.2rem;">
    <div style="font-size:1.15rem; font-weight:800; letter-spacing:-0.02em; color:#F5F5F7;">
        VERIF<span style="color:#39FF14;">I</span>AGE
    </div>
</div>
""", unsafe_allow_html=True)

st.components.v1.html("""
<script>
(function() {
    function highlightActiveNav() {
        const doc = window.parent.document;
        const links = doc.querySelectorAll('[data-testid="stSidebarNavLink"]');
        links.forEach(function(link) {
            const isActive = link.getAttribute('aria-current') === 'page';
            link.style.setProperty('background-color', isActive ? 'rgba(57,255,20,0.12)' : 'transparent', 'important');
            link.style.setProperty('border-left', isActive ? '2px solid #39FF14' : '2px solid transparent', 'important');
            const p = link.querySelector('p');
            if (p) p.style.setProperty('color', isActive ? '#39FF14' : '', 'important');
        });
    }
    highlightActiveNav();
    const doc = window.parent.document;
    const target = doc.querySelector('[data-testid="stSidebarNav"]') || doc.body;
    if (target && !target.__verifiageNavObserver) {
        const observer = new MutationObserver(highlightActiveNav);
        observer.observe(target, {subtree: true, attributes: true, childList: true, attributeFilter: ['aria-current', 'class']});
        target.__verifiageNavObserver = true;
    }
    setInterval(highlightActiveNav, 500);
})();
</script>
""", height=0)

pages = [
    st.Page("pages_src/dashboard.py", title="Dashboard", default=True),
    st.Page("pages_src/face_analysis.py", title="Face Analysis"),
    st.Page("pages_src/upload_analysis.py", title="Upload Analysis"),
    st.Page("pages_src/architecture.py", title="Architecture"),
    st.Page("pages_src/kpis.py", title="KPIs"),
    st.Page("pages_src/settings.py", title="Settings"),
    st.Page("pages_src/developer.py", title="Developer Mode"),
]

nav = st.navigation(pages, position="sidebar")
nav.run()
