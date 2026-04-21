from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Header, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..ai.inference_service import (
    load_disease_table,
    load_drug_table,
    load_links,
    load_protein_table,
    load_drug_protein_links,
    load_protein_disease_links,
    predict_diseases_by_drug_name,
    predict_drugs_by_disease_name,
)
from ..database import get_db
from ..models import Disease, Drug, DrugDiseaseLink, PredictionHistory, Protein, DrugProteinLink, ProteinDiseaseLink, User
from ..schemas import (
    DiseaseIn,
    DrugIn,
    HistoryItem,
    LinkIn,
    LoginRequest,
    LoginResponse,
    PredictRequest,
    PredictResponse,
    PredictionItem,
    StatsResponse,
)
from ..security import AuthUser, create_token, hash_password, verify_password

router = APIRouter(prefix="/api", tags=["api"])
TOKENS: dict[str, AuthUser] = {}


def bootstrap_data(db: Session) -> None:
    """Load dữ liệu từ tất cả 3 dataset (B, C, F) vào database với deduplication."""
    if db.query(User).count() == 0:
        db.add(User(username="admin", password_hash=hash_password("admin123"), role="admin"))
        db.add(User(username="user", password_hash=hash_password("user123"), role="user"))
        db.commit()

    # ── Load Drugs từ tất cả 3 dataset ──
    if db.query(Drug).count() == 0:
        drug_names_seen: set[str] = set()
        next_drug_id = 1
        for dataset in ["B-dataset", "C-dataset", "F-dataset"]:
            try:
                drugs = load_drug_table(dataset)
                for _, row in drugs.iterrows():
                    drug_name = str(row["name"])
                    if drug_name not in drug_names_seen:
                        db.add(Drug(id=next_drug_id, name=drug_name))
                        drug_names_seen.add(drug_name)
                        next_drug_id += 1
            except Exception:
                pass  # Skip nếu dataset không tồn tại
        db.commit()

    # ── Load Diseases từ tất cả 3 dataset ──
    if db.query(Disease).count() == 0:
        disease_names_seen: set[str] = set()
        next_disease_id = 1
        for dataset in ["B-dataset", "C-dataset", "F-dataset"]:
            try:
                diseases = load_disease_table(dataset)
                for _, row in diseases.iterrows():
                    disease_name = str(row["name"])
                    if disease_name not in disease_names_seen:
                        db.add(Disease(id=next_disease_id, name=disease_name))
                        disease_names_seen.add(disease_name)
                        next_disease_id += 1
            except Exception:
                pass
        db.commit()

    # ── Load Drug-Disease Links từ tất cả 3 dataset ──
    if db.query(DrugDiseaseLink).count() == 0:
        drug_name_map = {d.name: d.id for d in db.query(Drug).all()}
        disease_name_map = {d.name: d.id for d in db.query(Disease).all()}
        links_seen: set[tuple[int, int]] = set()
        for dataset in ["B-dataset", "C-dataset", "F-dataset"]:
            try:
                dataset_drugs = load_drug_table(dataset)
                dataset_diseases = load_disease_table(dataset)
                links = load_links(dataset)
                for _, row in links.iterrows():
                    drug_idx = int(row["drug"])
                    disease_idx = int(row["disease"])
                    if drug_idx < len(dataset_drugs) and disease_idx < len(dataset_diseases):
                        drug_name = str(dataset_drugs.iloc[drug_idx]["name"])
                        disease_name = str(dataset_diseases.iloc[disease_idx]["name"])
                        if drug_name in drug_name_map and disease_name in disease_name_map:
                            link_key = (drug_name_map[drug_name], disease_name_map[disease_name])
                            if link_key not in links_seen:
                                db.add(DrugDiseaseLink(drug_id=link_key[0], disease_id=link_key[1]))
                                links_seen.add(link_key)
            except Exception:
                pass
        db.commit()

    # ── Load Proteins từ tất cả 3 dataset ──
    if db.query(Protein).count() == 0:
        protein_accessions_seen: set[str] = set()
        next_protein_id = 1
        for dataset in ["B-dataset", "C-dataset", "F-dataset"]:
            try:
                proteins = load_protein_table(dataset)
                for _, row in proteins.iterrows():
                    accession = str(row["accession"])
                    if accession not in protein_accessions_seen:
                        db.add(Protein(id=next_protein_id, accession=accession, sequence=str(row.get("sequence", ""))))
                        protein_accessions_seen.add(accession)
                        next_protein_id += 1
            except Exception:
                pass
        db.commit()

    # ── Load Drug-Protein Links từ tất cả 3 dataset ──
    if db.query(DrugProteinLink).count() == 0:
        drug_name_map = {d.name: d.id for d in db.query(Drug).all()}
        protein_accession_map = {p.accession: p.id for p in db.query(Protein).all()}
        links_seen: set[tuple[int, int]] = set()
        for dataset in ["B-dataset", "C-dataset", "F-dataset"]:
            try:
                dataset_drugs = load_drug_table(dataset)
                dataset_proteins = load_protein_table(dataset)
                links = load_drug_protein_links(dataset)
                for _, row in links.iterrows():
                    drug_idx = int(row["drug"])
                    protein_idx = int(row["protein"])
                    if drug_idx < len(dataset_drugs) and protein_idx < len(dataset_proteins):
                        drug_name = str(dataset_drugs.iloc[drug_idx]["name"])
                        protein_accession = str(dataset_proteins.iloc[protein_idx]["accession"])
                        if drug_name in drug_name_map and protein_accession in protein_accession_map:
                            link_key = (drug_name_map[drug_name], protein_accession_map[protein_accession])
                            if link_key not in links_seen:
                                db.add(DrugProteinLink(drug_id=link_key[0], protein_id=link_key[1]))
                                links_seen.add(link_key)
            except Exception:
                pass
        db.commit()

    # ── Load Protein-Disease Links từ tất cả 3 dataset ──
    if db.query(ProteinDiseaseLink).count() == 0:
        disease_name_map = {d.name: d.id for d in db.query(Disease).all()}
        protein_accession_map = {p.accession: p.id for p in db.query(Protein).all()}
        links_seen: set[tuple[int, int]] = set()
        for dataset in ["B-dataset", "C-dataset", "F-dataset"]:
            try:
                dataset_diseases = load_disease_table(dataset)
                dataset_proteins = load_protein_table(dataset)
                links = load_protein_disease_links(dataset)
                for _, row in links.iterrows():
                    protein_idx = int(row["protein"])
                    disease_idx = int(row["disease"])
                    if protein_idx < len(dataset_proteins) and disease_idx < len(dataset_diseases):
                        protein_accession = str(dataset_proteins.iloc[protein_idx]["accession"])
                        disease_name = str(dataset_diseases.iloc[disease_idx]["name"])
                        if protein_accession in protein_accession_map and disease_name in disease_name_map:
                            link_key = (protein_accession_map[protein_accession], disease_name_map[disease_name])
                            if link_key not in links_seen:
                                db.add(ProteinDiseaseLink(protein_id=link_key[0], disease_id=link_key[1]))
                                links_seen.add(link_key)
            except Exception:
                pass
        db.commit()


def _auth_from_header(authorization: str | None) -> AuthUser:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Thieu token")
    token = authorization.split(" ", 1)[1].strip()
    user = TOKENS.get(token)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token khong hop le")
    return user


def get_current_user(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> AuthUser:
    return _auth_from_header(authorization)


def require_admin(user: Annotated[AuthUser, Depends(get_current_user)]) -> AuthUser:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Chi admin duoc phep")
    return user


@router.get("/health")
def health(db: Session = Depends(get_db)) -> dict:
    bootstrap_data(db)
    return {"status": "ok"}


@router.post("/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> LoginResponse:
    bootstrap_data(db)
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sai tai khoan/ mat khau")
    token = create_token()
    TOKENS[token] = AuthUser(id=user.id, username=user.username, role=user.role)
    return LoginResponse(token=token, username=user.username, role=user.role)


@router.post("/predict/drug-to-disease", response_model=PredictResponse)
def predict_drug_to_disease(
    payload: PredictRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
    db: Session = Depends(get_db),
) -> PredictResponse:
    input_name, preds = predict_diseases_by_drug_name(
        drug_name=payload.name,
        top_k=payload.top_k,
        threshold=payload.threshold,
    )
    for item in preds:
        db.add(
            PredictionHistory(
                user_id=user.id,
                direction="drug_to_disease",
                input_name=input_name,
                target_id=item["id"],
                target_name=item["name"],
                score=float(item["score"]),
                known=bool(item.get("known", False)),
            )
        )
    db.commit()
    return PredictResponse(
        direction="drug_to_disease",
        input_name=input_name,
        results=[PredictionItem(**x) for x in preds],
    )


@router.post("/predict/disease-to-drug", response_model=PredictResponse)
def predict_disease_to_drug(
    payload: PredictRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
    db: Session = Depends(get_db),
) -> PredictResponse:
    input_name, preds = predict_drugs_by_disease_name(
        disease_name=payload.name,
        top_k=payload.top_k,
        threshold=payload.threshold,
    )
    for item in preds:
        db.add(
            PredictionHistory(
                user_id=user.id,
                direction="disease_to_drug",
                input_name=input_name,
                target_id=item["id"],
                target_name=item["name"],
                score=float(item["score"]),
                known=bool(item.get("known", False)),
            )
        )
    db.commit()
    return PredictResponse(
        direction="disease_to_drug",
        input_name=input_name,
        results=[PredictionItem(**x) for x in preds],
    )


@router.get("/history", response_model=list[HistoryItem])
def history(user: Annotated[AuthUser, Depends(get_current_user)], db: Session = Depends(get_db)) -> list[HistoryItem]:
    rows = (
        db.query(PredictionHistory)
        .filter(PredictionHistory.user_id == user.id)
        .order_by(PredictionHistory.timestamp.desc())
        .limit(200)
        .all()
    )
    return [
        HistoryItem(
            id=r.id,
            direction=r.direction,
            input_name=r.input_name,
            target_id=r.target_id,
            target_name=r.target_name,
            score=r.score,
            known=bool(r.known),
            timestamp=r.timestamp,
        )
        for r in rows
    ]


@router.get("/drugs")
def list_drugs(
    user: AuthUser = Depends(get_current_user),
    limit: int = 200,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> list[dict]:
    _ = user
    rows = db.query(Drug).order_by(Drug.id.asc()).offset(max(0, offset)).limit(min(max(1, limit), 1000)).all()
    return [{"id": r.id, "name": r.name, "external_id": r.external_id, "smiles": r.smiles} for r in rows]


@router.get("/diseases")
def list_diseases(
    user: AuthUser = Depends(get_current_user),
    limit: int = 200,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> list[dict]:
    _ = user
    rows = db.query(Disease).order_by(Disease.id.asc()).offset(max(0, offset)).limit(min(max(1, limit), 1000)).all()
    return [{"id": r.id, "name": r.name} for r in rows]


@router.get("/proteins")
def list_proteins(
    user: AuthUser = Depends(get_current_user),
    limit: int = 200,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> list[dict]:
    _ = user
    rows = db.query(Protein).order_by(Protein.id.asc()).offset(max(0, offset)).limit(min(max(1, limit), 1000)).all()
    return [{"id": r.id, "name": r.accession, "sequence": r.sequence} for r in rows]


@router.get("/proteins/{protein_id}/links")
def get_protein_links(
    protein_id: int,
    user: AuthUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    _ = user
    protein = db.query(Protein).filter(Protein.id == protein_id).first()
    if not protein:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Protein không tồn tại")
    drug_rows = (
        db.query(Drug)
        .join(DrugProteinLink, Drug.id == DrugProteinLink.drug_id)
        .filter(DrugProteinLink.protein_id == protein_id)
        .order_by(Drug.id.asc())
        .all()
    )
    disease_rows = (
        db.query(Disease)
        .join(ProteinDiseaseLink, Disease.id == ProteinDiseaseLink.disease_id)
        .filter(ProteinDiseaseLink.protein_id == protein_id)
        .order_by(Disease.id.asc())
        .all()
    )
    return {
        "protein_id": protein.id,
        "accession": protein.accession,
        "sequence": protein.sequence,
        "drugs": [{"id": r.id, "name": r.name} for r in drug_rows],
        "diseases": [{"id": r.id, "name": r.name} for r in disease_rows],
    }


@router.get("/links")
def list_links(
    user: AuthUser = Depends(get_current_user),
    limit: int = 500,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> list[dict]:
    _ = user
    rows = (
        db.query(DrugDiseaseLink)
        .order_by(DrugDiseaseLink.id.asc())
        .offset(max(0, offset))
        .limit(min(max(1, limit), 2000))
        .all()
    )
    return [{"id": r.id, "drug_id": r.drug_id, "disease_id": r.disease_id} for r in rows]


@router.get("/admin/stats", response_model=StatsResponse)
def admin_stats(_: Annotated[AuthUser, Depends(require_admin)], db: Session = Depends(get_db)) -> StatsResponse:
    return StatsResponse(
        total_users=db.query(User).count(),
        total_drugs=db.query(Drug).count(),
        total_diseases=db.query(Disease).count(),
        total_links=db.query(DrugDiseaseLink).count(),
        total_predictions=db.query(PredictionHistory).count(),
    )


@router.get("/admin/stats/predictions-by-direction")
def admin_prediction_direction_stats(
    _: Annotated[AuthUser, Depends(require_admin)],
    db: Session = Depends(get_db),
) -> list[dict]:
    rows = (
        db.query(PredictionHistory.direction, func.count(PredictionHistory.id))
        .group_by(PredictionHistory.direction)
        .all()
    )
    return [{"direction": direction, "count": int(count)} for direction, count in rows]


@router.get("/admin/predictions")
def admin_list_predictions(
    _: Annotated[AuthUser, Depends(require_admin)],
    limit: int = 200,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> list[dict]:
    rows = (
        db.query(PredictionHistory)
        .order_by(PredictionHistory.timestamp.desc())
        .offset(max(0, offset))
        .limit(min(max(1, limit), 2000))
        .all()
    )
    return [
        {
            "id": r.id,
            "user_id": r.user_id,
            "direction": r.direction,
            "input_name": r.input_name,
            "target_id": r.target_id,
            "target_name": r.target_name,
            "score": r.score,
            "timestamp": r.timestamp.isoformat(),
        }
        for r in rows
    ]


@router.post("/admin/drugs")
def admin_create_drug(payload: DrugIn, _: Annotated[AuthUser, Depends(require_admin)], db: Session = Depends(get_db)) -> dict:
    db.merge(Drug(id=payload.id, name=payload.name, external_id=payload.external_id, smiles=payload.smiles))
    db.commit()
    return {"ok": True}


@router.post("/admin/diseases")
def admin_create_disease(payload: DiseaseIn, _: Annotated[AuthUser, Depends(require_admin)], db: Session = Depends(get_db)) -> dict:
    db.merge(Disease(id=payload.id, name=payload.name))
    db.commit()
    return {"ok": True}


@router.delete("/admin/drugs/{drug_id}")
def admin_delete_drug(drug_id: int, _: Annotated[AuthUser, Depends(require_admin)], db: Session = Depends(get_db)) -> dict:
    db.query(DrugDiseaseLink).filter(DrugDiseaseLink.drug_id == drug_id).delete()
    db.query(Drug).filter(Drug.id == drug_id).delete()
    db.commit()
    return {"ok": True}


@router.delete("/admin/diseases/{disease_id}")
def admin_delete_disease(disease_id: int, _: Annotated[AuthUser, Depends(require_admin)], db: Session = Depends(get_db)) -> dict:
    db.query(DrugDiseaseLink).filter(DrugDiseaseLink.disease_id == disease_id).delete()
    db.query(Disease).filter(Disease.id == disease_id).delete()
    db.commit()
    return {"ok": True}


@router.post("/admin/links")
def admin_create_link(payload: LinkIn, _: Annotated[AuthUser, Depends(require_admin)], db: Session = Depends(get_db)) -> dict:
    exists = (
        db.query(DrugDiseaseLink)
        .filter(DrugDiseaseLink.drug_id == payload.drug_id, DrugDiseaseLink.disease_id == payload.disease_id)
        .first()
    )
    if not exists:
        db.add(DrugDiseaseLink(drug_id=payload.drug_id, disease_id=payload.disease_id))
        db.commit()
    return {"ok": True}


@router.delete("/admin/links/{drug_id}/{disease_id}")
def admin_delete_link(
    drug_id: int,
    disease_id: int,
    _: Annotated[AuthUser, Depends(require_admin)],
    db: Session = Depends(get_db),
) -> dict:
    db.query(DrugDiseaseLink).filter(
        DrugDiseaseLink.drug_id == drug_id,
        DrugDiseaseLink.disease_id == disease_id,
    ).delete()
    db.commit()
    return {"ok": True}
