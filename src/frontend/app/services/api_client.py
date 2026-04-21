from __future__ import annotations

from typing import Any

import requests


class ApiClient:
    def __init__(self, base_url: str, token: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: int = 60,
    ) -> Any:
        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        url = f"{self.base_url}{path}"
        response = requests.request(method=method, url=url, headers=headers, json=payload, timeout=timeout)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(f"{response.status_code}: {detail}")
        if not response.text:
            return {}
        try:
            return response.json()
        except ValueError:
            return {}

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def login(self, username: str, password: str) -> dict[str, Any]:
        return self._request("POST", "/auth/login", payload={"username": username, "password": password})

    def predict_drug_to_disease(self, name: str, top_k: int, threshold: float) -> dict[str, Any]:
        return self._request(
            "POST",
            "/predict/drug-to-disease",
            payload={
                "name": str(name),
                "top_k": int(top_k),
                "threshold": float(threshold),
            },
        )

    def predict_disease_to_drug(self, name: str, top_k: int, threshold: float) -> dict[str, Any]:
        return self._request(
            "POST",
            "/predict/disease-to-drug",
            payload={
                "name": str(name),
                "top_k": int(top_k),
                "threshold": float(threshold),
            },
        )

    def history(self) -> list[dict[str, Any]]:
        return self._request("GET", "/history")

    def list_drugs(self, limit: int = 200) -> list[dict[str, Any]]:
        return self._request("GET", f"/drugs?limit={int(limit)}")

    def list_diseases(self, limit: int = 200) -> list[dict[str, Any]]:
        return self._request("GET", f"/diseases?limit={int(limit)}")

    def list_proteins(self, limit: int = 200) -> list[dict[str, Any]]:
        return self._request("GET", f"/proteins?limit={int(limit)}")

    def get_protein_links(self, protein_id: int) -> dict[str, Any]:
        return self._request("GET", f"/proteins/{int(protein_id)}/links")

    def list_links(self, limit: int = 300) -> list[dict[str, Any]]:
        return self._request("GET", f"/links?limit={int(limit)}")

    def admin_stats(self) -> dict[str, Any]:
        return self._request("GET", "/admin/stats")

    def admin_prediction_direction_stats(self) -> list[dict[str, Any]]:
        return self._request("GET", "/admin/stats/predictions-by-direction")

    def admin_predictions(self, limit: int = 300) -> list[dict[str, Any]]:
        return self._request("GET", f"/admin/predictions?limit={int(limit)}")

    def admin_save_drug(self, drug_id: int, name: str, external_id: str | None, smiles: str | None) -> dict[str, Any]:
        return self._request(
            "POST",
            "/admin/drugs",
            payload={"id": int(drug_id), "name": str(name), "external_id": external_id, "smiles": smiles},
        )

    def admin_save_disease(self, disease_id: int, name: str) -> dict[str, Any]:
        return self._request("POST", "/admin/diseases", payload={"id": int(disease_id), "name": str(name)})

    def admin_save_link(self, drug_id: int, disease_id: int) -> dict[str, Any]:
        return self._request("POST", "/admin/links", payload={"drug_id": int(drug_id), "disease_id": int(disease_id)})
