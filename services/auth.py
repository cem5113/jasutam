# services/auth.py
from __future__ import annotations
import os
from typing import Literal, Optional

Role = Literal["Amir", "Kullanıcı"]

_ROLES: tuple[Role, Role] = ("Amir", "Kullanıcı")
_ROLE_ORDER = {"Kullanıcı": 0, "Amir": 1}

# Yaygın takma adlar → normalize
_ALIAS = {
    "amir": "Amir",
    "admin": "Amir",
    "manager": "Amir",
    "supervisor": "Amir",
    "kullanıcı": "Kullanıcı",
    "kullanici": "Kullanıcı",
    "user": "Kullanıcı",
    "viewer": "Kullanıcı",
}

def _normalize(value: Optional[str]) -> Role:
    if not value:
        return "Kullanıcı"
    v = str(value).strip()
    if v in _ROLES:
        return v  # type: ignore
    vlow = v.lower()
    if vlow in __ALIAS:
        return _ALIAS[vlow]  # type: ignore
    return "Kullanıcı"

def _query_param_role() -> Optional[str]:
    """Streamlit varsa ?role=… parametresini yakalamayı dener."""
    try:
        import streamlit as st
        # Yeni API: st.query_params; eski: st.experimental_get_query_params
        qp = getattr(st, "query_params", None)
        if qp is not None:
            return qp.get("role")
        qpe = getattr(st, "experimental_get_query_params", None)
        if callable(qpe):
            q = qpe() or {}
            val = q.get("role")
            if isinstance(val, list):
                return val[0]
            return val
    except Exception:
        pass
    return None

def get_role() -> Role:
    """
    Öncelik: session_state → URL ?role=… → ENV(APP_ROLE) → secrets.DEFAULT_ROLE → 'Kullanıcı'
    """
    # 1) Session
    try:
        import streamlit as st
        r = st.session_state.get("role")
        if r in _ROLES:
            return r  # type: ignore
    except Exception:
        pass

    # 2) Query param
    qp = _query_param_role()
    if qp:
        return _normalize(qp)

    # 3) ENV
    r_env = os.environ.get("APP_ROLE")
    if r_env:
        return _normalize(r_env)

    # 4) Secrets (varsayılan)
    try:
        import streamlit as st
        r_sec = st.secrets.get("DEFAULT_ROLE", None)  # örn: "Amir" / "Kullanıcı"
        if r_sec:
            return _normalize(r_sec)
    except Exception:
        pass

    # 5) Fallback
    return "Kullanıcı"

def set_role(role: Role) -> None:
    """Rolü hem session’a hem ENV’e yazar (UI/CLI uyumlu)."""
    try:
        import streamlit as st
        st.session_state["role"] = _normalize(role)
    except Exception:
        pass
    os.environ["APP_ROLE"] = _normalize(role)

def can_approve() -> bool:
    return get_role() == "Amir"

def has_permission(action: str) -> bool:
    """
    Basit izin matrisi. Gerekirse genişlet:
      approve → Amir
      view    → Kullanıcı+
    """
    role = get_role()
    if action == "approve":
        return role == "Amir"
    # varsayılan: görüntüleme serbest
    return True

def ensure_amir_or_warn(action_name: str = "bu işlem") -> bool:
    """
    Streamlit içinde çağrılırsa uygun uyarıyı basar; CLI’da sadece False döner.
    """
    if can_approve():
        return True
    try:
        import streamlit as st
        st.warning(f"🔒 {action_name} için **Amir** rolü gerekli.")
    except Exception:
        pass
    return False

def role_selector_in_sidebar(default: Role = "Kullanıcı") -> Role:
    """
    Streamlit varsa sidebar’da rol seçici render eder; seçimi döndürür.
    Query param (?role=Amir) varsa ilk yüklemede onu uygular.
    """
    cur = get_role() or default
    try:
        import streamlit as st
        # İlk yüklemede URL paramı geldiyse session’a yaz
        qp_role = _query_param_role()
        if qp_role:
            cur = _normalize(qp_role)
            st.session_state["role"] = cur

        role = st.sidebar.selectbox("Rol", _ROLES, index=_ROLES.index(cur))
        set_role(role)
        st.sidebar.caption("Rol: **Amir** → onay/yetki • **Kullanıcı** → görüntüleme.")
        return role  # type: ignore
    except Exception:
        # Streamlit yoksa mevcut rolü döndür
        return _normalize(cur)
