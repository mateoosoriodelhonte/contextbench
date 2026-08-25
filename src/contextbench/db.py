"""SQLite engine and repository helpers."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from .models import Base, Document, Project


def create_session_factory(path: str | Path = "contextbench.sqlite3") -> sessionmaker[Session]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def session_scope(factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class Repository:
    def __init__(self, factory: sessionmaker[Session]) -> None:
        self.factory = factory

    def create_project(self, name: str, description: str | None = None) -> Project:
        with self.factory.begin() as session:
            project = Project(name=name, description=description)
            session.add(project)
            session.flush()
            session.refresh(project)
            return project

    def get_project(self, project_id: str) -> Project | None:
        with self.factory() as session:
            return session.get(Project, project_id)

    def list_projects(self) -> list[Project]:
        with self.factory() as session:
            return list(session.scalars(select(Project).order_by(Project.created_at.desc())))

    def add_document(self, document: Document) -> Document:
        with self.factory.begin() as session:
            session.add(document)
            session.flush()
            session.refresh(document)
            return document
