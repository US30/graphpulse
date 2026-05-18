import os
from celery import Celery

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")


def create_celery_app() -> Celery:
    """Factory that creates and configures the Celery application."""
    app = Celery(
        "graphpulse",
        broker=CELERY_BROKER_URL,
        backend=CELERY_RESULT_BACKEND,
        include=[
            "services.tasks",
        ],
    )

    app.conf.update(
        # Reliability
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        # Serialisation
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        # Routing
        task_routes={
            "services.tasks.retrain_model": {"queue": "fraud"},
            "services.tasks.run_drift_check": {"queue": "fraud"},
            "services.tasks.score_batch_async": {"queue": "fraud"},
        },
        # Timeouts (seconds)
        task_soft_time_limit=300,
        task_time_limit=600,
        # Result expiry
        result_expires=86400,  # 24 h
        # Timezone
        timezone="UTC",
        enable_utc=True,
    )

    return app


celery_app = create_celery_app()
