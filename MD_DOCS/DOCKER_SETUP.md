# Docker Setup Guide

This guide explains how to run the ET Social Intelligence System using Docker for consistent, reproducible deployments.

## Prerequisites

- Docker (version 20.10+)
- Docker Compose (version 1.29+)

## Quick Start

### 1. Build the Docker Image

```bash
docker-compose build
```

This will:
- Install all Python dependencies
- Download spaCy models (`en_core_web_lg`)
- Pre-configure the environment

### 2. Run Batch Processing

Process all unprocessed CSV files:

```bash
docker-compose run --rm et-intel python -m et_intel.cli.cli --batch
```

### 3. Start the Dashboard

Launch the Streamlit dashboard:

```bash
docker-compose up dashboard
```

Access at: http://localhost:8501

## Docker Commands Reference

### Process a Single CSV File

```bash
docker-compose run --rm et-intel python -m et_intel.cli.cli \
  --import /app/et_intel/data/uploads/comments.csv \
  --platform instagram \
  --subject "Taylor Swift"
```

### Generate Intelligence Brief

```bash
docker-compose run --rm et-intel python -m et_intel.cli.cli \
  --generate \
  --last-30-days
```

### View Database Stats

```bash
docker-compose run --rm et-intel python -m et_intel.cli.cli --stats
```

### Run Tests

```bash
docker-compose run --rm et-intel pytest tests/ -v
```

## Data Persistence

The `docker-compose.yml` mounts the following directories as volumes:

- `./et_intel/data` - CSV uploads, processed files, database
- `./et_intel/reports` - Generated PDF reports and charts
- `./.env` - Environment variables (read-only)

**Important:** These directories are mounted as volumes, so data persists between container restarts.

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-key-here
LOG_LEVEL=INFO
```

The `.env` file is mounted read-only into the container.

## Troubleshooting

### Container won't start

Check logs:
```bash
docker-compose logs et-intel
```

### Database locked errors

If you see SQLite locking errors, ensure only one container is accessing the database at a time. SQLite doesn't handle concurrent writes well.

### Missing spaCy model

The Dockerfile downloads spaCy models during build. If you see errors, rebuild:

```bash
docker-compose build --no-cache
```

### Permission errors

On Linux/Mac, you may need to fix permissions:

```bash
sudo chown -R $USER:$USER et_intel/data et_intel/reports
```

## Production Deployment

For production, consider:

1. **Use PostgreSQL instead of SQLite** for better concurrency
2. **Add health checks** to docker-compose.yml
3. **Set up volume backups** for data persistence
4. **Use Docker secrets** for API keys instead of .env files
5. **Enable log rotation** for log files

## Development vs Production

### Development

```bash
# Build with cache
docker-compose build

# Run with live code mounting (if needed)
docker-compose run --rm -v $(pwd):/app et-intel python -m et_intel.cli.cli --batch
```

### Production

```bash
# Build production image
docker build -t et-intel:latest .

# Run with production settings
docker run -d \
  --name et-intel \
  -v $(pwd)/et_intel/data:/app/et_intel/data \
  -v $(pwd)/et_intel/reports:/app/et_intel/reports \
  --env-file .env.production \
  et-intel:latest \
  python -m et_intel.cli.cli --batch
```

