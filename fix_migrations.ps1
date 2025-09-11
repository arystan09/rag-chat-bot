# PowerShell script to fix Alembic migration conflicts
# Usage: .\fix_migrations.ps1

Write-Host "🔧 Fixing Alembic migration conflicts..." -ForegroundColor Yellow

# Check if there are multiple heads
Write-Host "📋 Checking current migration heads..." -ForegroundColor Cyan
$headsOutput = docker exec rag_app alembic heads
Write-Host $headsOutput

# Count the number of heads (excluding INFO lines)
$heads = $headsOutput | Where-Object { $_ -notmatch "INFO" -and $_ -notmatch "\[" } | Where-Object { $_.Trim() -ne "" }

if ($heads.Count -gt 1) {
    Write-Host "⚠️  Multiple heads detected: $($heads -join ', ')" -ForegroundColor Red
    Write-Host "🔄 Creating merge migration..." -ForegroundColor Yellow
    
    # Create merge migration
    $headsString = $heads -join " "
    docker exec rag_app alembic merge -m "merge heads" $headsString
    
    Write-Host "📈 Applying merge migration..." -ForegroundColor Yellow
    docker exec rag_app alembic upgrade head
    
    Write-Host "✅ Migration conflicts resolved!" -ForegroundColor Green
} else {
    Write-Host "✅ No migration conflicts detected." -ForegroundColor Green
}

Write-Host "📊 Current migration status:" -ForegroundColor Cyan
docker exec rag_app alembic current


