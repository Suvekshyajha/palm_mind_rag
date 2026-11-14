from datetime import date, datetime, time

from sqlalchemy import Date, DateTime, Integer, String, Time, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Document(Base):
    """One row per uploaded file."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[str] = mapped_column(String(10))
    file_path: Mapped[str] = mapped_column(String(500))
    chunking_strategy: Mapped[str] = mapped_column(String(20))
    chunk_count: Mapped[int] = mapped_column(Integer)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Booking(Base):
    """One row per completed interview booking (used in part 2)."""

    __tablename__ = "bookings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str] = mapped_column(String(255))
    interview_date: Mapped[date] = mapped_column(Date)
    interview_time: Mapped[time] = mapped_column(Time)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


async def save_document(session: AsyncSession, document: Document) -> None:
    session.add(document)
    await session.commit()


async def save_booking(session: AsyncSession, booking: Booking) -> None:
    session.add(booking)
    await session.commit()