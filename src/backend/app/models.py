from __future__ import annotations

from datetime import datetime
from typing import List

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default="user", nullable=False)

    predictions: Mapped[List[PredictionHistory]] = relationship(
        "PredictionHistory",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class Drug(Base):
    __tablename__ = "drugs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    external_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    smiles: Mapped[str | None] = mapped_column(Text, nullable=True)
    features: Mapped[str | None] = mapped_column(Text, nullable=True)


class Disease(Base):
    __tablename__ = "diseases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    features: Mapped[str | None] = mapped_column(Text, nullable=True)


class DrugDiseaseLink(Base):
    __tablename__ = "drug_disease_links"
    __table_args__ = (UniqueConstraint("drug_id", "disease_id", name="uq_drug_disease"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    drug_id: Mapped[int] = mapped_column(ForeignKey("drugs.id"), nullable=False, index=True)
    disease_id: Mapped[int] = mapped_column(ForeignKey("diseases.id"), nullable=False, index=True)


class Protein(Base):
    __tablename__ = "proteins"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    accession: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    sequence: Mapped[str | None] = mapped_column(Text, nullable=True)


class DrugProteinLink(Base):
    __tablename__ = "drug_protein_links"
    __table_args__ = (UniqueConstraint("drug_id", "protein_id", name="uq_drug_protein"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    drug_id: Mapped[int] = mapped_column(ForeignKey("drugs.id"), nullable=False, index=True)
    protein_id: Mapped[int] = mapped_column(ForeignKey("proteins.id"), nullable=False, index=True)


class ProteinDiseaseLink(Base):
    __tablename__ = "protein_disease_links"
    __table_args__ = (UniqueConstraint("protein_id", "disease_id", name="uq_protein_disease"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    protein_id: Mapped[int] = mapped_column(ForeignKey("proteins.id"), nullable=False, index=True)
    disease_id: Mapped[int] = mapped_column(ForeignKey("diseases.id"), nullable=False, index=True)


class PredictionHistory(Base):
    __tablename__ = "predictions_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String(32), nullable=False)  # drug_to_disease | disease_to_drug
    input_name: Mapped[str] = mapped_column(String(255), nullable=False)
    target_id: Mapped[int] = mapped_column(Integer, nullable=False)
    target_name: Mapped[str] = mapped_column(String(255), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    known: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, server_default="0")
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user: Mapped[User] = relationship("User", back_populates="predictions")
