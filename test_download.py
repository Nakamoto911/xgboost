import streamlit as st
import base64
import streamlit.components.v1 as components

if st.button("Export to PDF"):
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n5 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 24 Tf\n100 700 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n0000000216 00000 n\n0000000304 00000 n\ntrailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n397\n%%EOF"
    
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f"""
    <a href="data:application/pdf;base64,{b64}" download="test.pdf" id="download_link"></a>
    <script>
        document.getElementById("download_link").click();
    </script>
    """
    components.html(href, height=0)
    st.success("Downloaded!")
