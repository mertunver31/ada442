import streamlit as st
import traceback
import sys

try:
    import app
except Exception as e:
    st.error("### Uygulama yüklenirken hata oluştu")
    st.error(f"**Hata mesajı:** {str(e)}")
    st.code(traceback.format_exc())
    st.info("Bu hata mesajını kopyalayıp GitHub'da bir issue olarak paylaşabilirsiniz.") 