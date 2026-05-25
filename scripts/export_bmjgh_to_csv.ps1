$ErrorActionPreference = 'Stop'

$root = 'C:\Users\adamk\Documents\New project\downloads\bmjgh_sources'
$rawDir = Join-Path $root 'raw'
$extractedDir = Join-Path $root 'extracted'
$csvDir = Join-Path $root 'csv'
$manifestPath = Join-Path $root 'csv_manifest.json'

New-Item -ItemType Directory -Force -Path $csvDir | Out-Null

function Get-ExcelConnectionString {
    param([Parameter(Mandatory = $true)][string]$Path)

    $ext = [System.IO.Path]::GetExtension($Path).ToLowerInvariant()
    switch ($ext) {
        '.xlsx' { return "Provider=Microsoft.ACE.OLEDB.12.0;Data Source=$Path;Extended Properties='Excel 12.0 Xml;HDR=YES;IMEX=1';" }
        '.xls' { return "Provider=Microsoft.ACE.OLEDB.12.0;Data Source=$Path;Extended Properties='Excel 8.0;HDR=YES;IMEX=1';" }
        default { throw "Unsupported Excel extension: $ext" }
    }
}

function Get-SheetNames {
    param([Parameter(Mandatory = $true)][string]$Path)

    $conn = New-Object System.Data.OleDb.OleDbConnection((Get-ExcelConnectionString -Path $Path))
    try {
        $conn.Open()
        $schema = $conn.GetOleDbSchemaTable([System.Data.OleDb.OleDbSchemaGuid]::Tables, $null)
        foreach ($row in $schema.Rows) {
            $tableName = [string]$row.TABLE_NAME
            if (-not $tableName) { continue }

            $clean = $tableName.Trim("'")
            if ($clean -notmatch '\$$') { continue }
            $clean
        }
    }
    finally {
        if ($conn.State -eq 'Open') { $conn.Close() }
    }
}

function Sanitize-Name {
    param([Parameter(Mandatory = $true)][string]$Name)

    $invalidChars = [System.IO.Path]::GetInvalidFileNameChars()
    $sanitized = $Name.Trim()
    foreach ($char in $invalidChars) {
        $sanitized = $sanitized.Replace($char, '_')
    }
    $sanitized = $sanitized -replace '\$', ''
    $sanitized = $sanitized -replace '\s+', '_'
    $sanitized = $sanitized -replace '_+', '_'
    $sanitized.Trim('_')
}

function Export-WorkbookToCsv {
    param([Parameter(Mandatory = $true)][string]$Path)

    $workbookName = [System.IO.Path]::GetFileNameWithoutExtension($Path)
    $targetDir = Join-Path $csvDir (Sanitize-Name -Name $workbookName)
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

    $conn = New-Object System.Data.OleDb.OleDbConnection((Get-ExcelConnectionString -Path $Path))
    $sheetResults = @()

    try {
        $conn.Open()
        $sheetNames = Get-SheetNames -Path $Path

        foreach ($sheetName in $sheetNames) {
            $query = "SELECT * FROM [$sheetName]"
            $adapter = New-Object System.Data.OleDb.OleDbDataAdapter($query, $conn)
            $table = New-Object System.Data.DataTable
            [void]$adapter.Fill($table)

            $sheetBase = Sanitize-Name -Name $sheetName
            if ([string]::IsNullOrWhiteSpace($sheetBase)) {
                $sheetBase = 'sheet'
            }

            $outPath = Join-Path $targetDir ($sheetBase + '.csv')
            $table | Export-Csv -LiteralPath $outPath -NoTypeInformation -Encoding UTF8

            $sheetResults += [PSCustomObject]@{
                sheet = $sheetName
                csvPath = $outPath
                rowCount = $table.Rows.Count
                columnCount = $table.Columns.Count
            }
        }
    }
    finally {
        if ($conn.State -eq 'Open') { $conn.Close() }
    }

    [PSCustomObject]@{
        sourcePath = $Path
        targetDir = $targetDir
        sheets = $sheetResults
    }
}

$excelFiles = Get-ChildItem -LiteralPath $rawDir -File |
    Where-Object { $_.Extension -in @('.xlsx', '.xls') } |
    Sort-Object Name

$results = @()
foreach ($file in $excelFiles) {
    $results += Export-WorkbookToCsv -Path $file.FullName
}

# Copy original CSV files from extracted datasets into the same CSV root.
$existingCsvs = Get-ChildItem -LiteralPath $extractedDir -Recurse -File -Filter *.csv -ErrorAction SilentlyContinue
foreach ($csv in $existingCsvs) {
    $datasetName = Split-Path -Leaf (Split-Path -Parent $csv.FullName)
    $targetDir = Join-Path $csvDir (Sanitize-Name -Name $datasetName)
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

    $targetPath = Join-Path $targetDir $csv.Name
    Copy-Item -LiteralPath $csv.FullName -Destination $targetPath -Force

    $results += [PSCustomObject]@{
        sourcePath = $csv.FullName
        targetDir = $targetDir
        sheets = @(
            [PSCustomObject]@{
                sheet = $null
                csvPath = $targetPath
                rowCount = $null
                columnCount = $null
            }
        )
    }
}

$results | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

$results |
    ForEach-Object {
        foreach ($sheet in $_.sheets) {
            [PSCustomObject]@{
                source = $_.sourcePath
                csvPath = $sheet.csvPath
                rows = $sheet.rowCount
                cols = $sheet.columnCount
            }
        }
    } |
    Sort-Object csvPath |
    Format-Table -AutoSize
