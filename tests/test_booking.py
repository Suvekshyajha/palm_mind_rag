from datetime import date, timedelta

from app.services.booking import validate_details


def _tomorrow() -> str:
    return (date.today() + timedelta(days=1)).isoformat()


def test_good_details_pass() -> None:
    details = {"name": "Sera Shrestha", "email": "sera@gmail.com", "date": _tomorrow(), "time": "10:30"}
    valid, problems = validate_details(details)
    assert problems == []
    assert set(valid) == {"name", "email", "date", "time"}


def test_bad_email_is_dropped_and_reported() -> None:
    valid, problems = validate_details({"email": "not-an-email"})
    assert "email" not in valid
    assert problems


def test_past_date_is_rejected() -> None:
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    valid, problems = validate_details({"date": yesterday})
    assert "date" not in valid
    assert problems