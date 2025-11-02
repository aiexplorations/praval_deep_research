#!/bin/bash

# Docker management script for Agentic Deep Research
# Usage: ./scripts/docker-manage.sh [command]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project configuration
PROJECT_NAME="agentic-deep-research"
COMPOSE_FILE="docker-compose.yml"

# Helper functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi

    # Check if docker-compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
}

create_env_file() {
    if [ ! -f .env ]; then
        print_info "Creating .env file from template..."
        cp .env.example .env
        print_warning "Please update .env file with your actual configuration values"
        return 1
    fi
    return 0
}

# Command functions
start_services() {
    print_info "Starting core research infrastructure..."
    
    create_env_file
    
    # Start core services
    docker-compose up -d rabbitmq qdrant minio redis
    
    print_info "Waiting for services to be healthy..."
    
    # Wait for services to be healthy
    services=("rabbitmq" "qdrant" "minio" "redis")
    for service in "${services[@]}"; do
        print_info "Waiting for $service to be healthy..."
        timeout 120 bash -c "until docker-compose exec $service echo 'Service is up' 2>/dev/null; do sleep 2; done" || {
            print_error "Failed to start $service within timeout"
            return 1
        }
    done
    
    print_success "Core infrastructure is running!"
    print_info "Services available at:"
    echo "  - RabbitMQ Management: http://localhost:15672 (user: research_user, pass: research_pass)"
    echo "  - Qdrant API: http://localhost:6333"
    echo "  - MinIO Console: http://localhost:9001 (user: minioadmin, pass: minioadmin)"
    echo "  - Redis: localhost:6379"
}

start_with_postgres() {
    print_info "Starting infrastructure with PostgreSQL..."
    create_env_file
    docker-compose --profile with-postgres up -d
    print_success "Infrastructure with PostgreSQL is running!"
}

start_with_monitoring() {
    print_info "Starting infrastructure with monitoring..."
    create_env_file
    docker-compose --profile monitoring up -d
    print_success "Infrastructure with monitoring is running!"
    print_info "Prometheus available at: http://localhost:9090"
}

start_dev_tools() {
    print_info "Starting development tools..."
    docker-compose --profile dev-tools up -d
    print_success "Development tools are running!"
    print_info "Adminer available at: http://localhost:8080"
}

stop_services() {
    print_info "Stopping all services..."
    docker-compose down
    print_success "All services stopped!"
}

restart_services() {
    print_info "Restarting services..."
    docker-compose restart
    print_success "Services restarted!"
}

show_status() {
    print_info "Service status:"
    docker-compose ps
}

show_logs() {
    local service=${1:-}
    if [ -n "$service" ]; then
        print_info "Showing logs for $service..."
        docker-compose logs -f "$service"
    else
        print_info "Showing logs for all services..."
        docker-compose logs -f
    fi
}

cleanup() {
    print_warning "This will remove all containers, networks, and volumes!"
    read -p "Are you sure? (y/N): " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cleaning up..."
        docker-compose down -v --remove-orphans
        docker system prune -f
        print_success "Cleanup completed!"
    else
        print_info "Cleanup cancelled."
    fi
}

reset_data() {
    print_warning "This will remove all data volumes!"
    read -p "Are you sure? (y/N): " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Resetting data..."
        docker-compose down
        docker volume rm $(docker volume ls -q | grep "${PROJECT_NAME}" | tr '\n' ' ') 2>/dev/null || true
        print_success "Data reset completed!"
    else
        print_info "Reset cancelled."
    fi
}

backup_data() {
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    print_info "Creating data backup..."
    
    # Backup PostgreSQL if running
    if docker-compose ps postgres | grep -q "Up"; then
        print_info "Backing up PostgreSQL..."
        docker-compose exec postgres pg_dump -U research_user research_metadata > "$backup_dir/postgres_backup.sql"
    fi
    
    # Backup Redis if running
    if docker-compose ps redis | grep -q "Up"; then
        print_info "Backing up Redis..."
        docker-compose exec redis redis-cli BGSAVE
        docker cp $(docker-compose ps -q redis):/data/dump.rdb "$backup_dir/redis_backup.rdb"
    fi
    
    print_success "Backup completed in $backup_dir"
}

health_check() {
    print_info "Performing health check..."
    
    services=("rabbitmq:5672" "qdrant:6333" "minio:9000" "redis:6379")
    all_healthy=true
    
    for service in "${services[@]}"; do
        service_name=$(echo $service | cut -d':' -f1)
        port=$(echo $service | cut -d':' -f2)
        
        if nc -z localhost $port 2>/dev/null; then
            print_success "$service_name is healthy"
        else
            print_error "$service_name is not responding"
            all_healthy=false
        fi
    done
    
    if $all_healthy; then
        print_success "All services are healthy!"
        return 0
    else
        print_error "Some services are not healthy"
        return 1
    fi
}

show_help() {
    echo "Agentic Deep Research - Docker Management Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start              Start core infrastructure (RabbitMQ, Qdrant, MinIO, Redis)"
    echo "  start-postgres     Start infrastructure with PostgreSQL"
    echo "  start-monitoring   Start infrastructure with Prometheus monitoring"
    echo "  start-dev          Start development tools (Adminer)"
    echo "  stop               Stop all services"
    echo "  restart            Restart all services"
    echo "  status             Show service status"
    echo "  logs [service]     Show logs (optional: for specific service)"
    echo "  health             Check service health"
    echo "  backup             Create data backup"
    echo "  reset              Reset all data (removes volumes)"
    echo "  cleanup            Remove all containers, networks, and volumes"
    echo "  help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 logs rabbitmq"
    echo "  $0 health"
}

# Main script logic
main() {
    check_requirements
    
    case "${1:-help}" in
        "start")
            start_services
            ;;
        "start-postgres")
            start_with_postgres
            ;;
        "start-monitoring")
            start_with_monitoring
            ;;
        "start-dev")
            start_dev_tools
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            restart_services
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs "${2:-}"
            ;;
        "health")
            health_check
            ;;
        "backup")
            backup_data
            ;;
        "reset")
            reset_data
            ;;
        "cleanup")
            cleanup
            ;;
        "help")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"