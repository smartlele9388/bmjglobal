$ErrorActionPreference = 'Stop'

$root = 'C:\Users\adamk\Documents\New project\downloads\bmjgh_sources'
$auxRoot = Join-Path $root 'aux'
$years = 2016..2024

New-Item -ItemType Directory -Force -Path $auxRoot | Out-Null

$whoUrl = 'https://ghoapi.azureedge.net/api/DIMENSION/COUNTRY/DimensionValues?$top=500'
$whoJsonPath = Join-Path $auxRoot 'who_country_dimensionvalues.json'
$whoCsvPath = Join-Path $auxRoot 'who_country_region_mapping.csv'

$climateTimeseriesUrl = 'https://cckpapi.worldbank.org/cckp/v1/cru-x0.5_timeseries_tas_timeseries_annual_1901-2023_mean_historical_cru_ts4.08_mean/all_countries?_format=json'
$climateBaselineUrl = 'https://cckpapi.worldbank.org/cckp/v1/cru-x0.5_climatology_tas_climatology_annual_1995-2014_mean_historical_cru_ts4.08_mean/all_countries?_format=json'
$climateTimeseriesJsonPath = Join-Path $auxRoot 'cckp_cru_tas_timeseries_annual_1901_2023.json'
$climateBaselineJsonPath = Join-Path $auxRoot 'cckp_cru_tas_climatology_1995_2014.json'
$climateCsvPath = Join-Path $auxRoot 'cckp_cru_tas_anomaly_2016_2024.csv'
$manifestPath = Join-Path $auxRoot 'aux_manifest.json'

function Save-Json {
    param(
        [Parameter(Mandatory = $true)]$Object,
        [Parameter(Mandatory = $true)][string]$Path
    )

    $Object | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Get-FirstNumericValue {
    param($ValueObject)

    if ($null -eq $ValueObject) { return $null }
    foreach ($property in $ValueObject.PSObject.Properties) {
        if ($null -ne $property.Value) {
            return [double]$property.Value
        }
    }

    return $null
}

$whoResponse = Invoke-RestMethod -Uri $whoUrl -TimeoutSec 120
Save-Json -Object $whoResponse -Path $whoJsonPath

$whoRows = foreach ($row in $whoResponse.value) {
    if ($row.Dimension -ne 'COUNTRY' -or $row.ParentDimension -ne 'REGION') { continue }

    $regionCode = $row.ParentCode
    $regionName = $row.ParentTitle
    $note = 'WHO GHO country dimension values'

    if ($row.Code -eq 'IDN') {
        # WHO reassigned Indonesia to WPR in 2025; the analysis panel ends in 2024.
        $regionCode = 'SEAR'
        $regionName = 'South-East Asia'
        $note = 'WHO GHO country dimension values with Indonesia reset to its pre-2025 WHO region'
    }

    [PSCustomObject][ordered]@{
        code = $row.Code
        country_name = $row.Title
        who_region_code = $regionCode
        who_region = $regionName
        mapping_note = $note
    }
}

$whoRows |
    Sort-Object code |
    Export-Csv -LiteralPath $whoCsvPath -NoTypeInformation -Encoding UTF8

$climateTimeseriesResponse = Invoke-RestMethod -Uri $climateTimeseriesUrl -TimeoutSec 300
$climateBaselineResponse = Invoke-RestMethod -Uri $climateBaselineUrl -TimeoutSec 120

Save-Json -Object $climateTimeseriesResponse -Path $climateTimeseriesJsonPath
Save-Json -Object $climateBaselineResponse -Path $climateBaselineJsonPath

$climateBaselineByCode = @{}
foreach ($property in $climateBaselineResponse.data.PSObject.Properties) {
    $climateBaselineByCode[$property.Name] = Get-FirstNumericValue -ValueObject $property.Value
}

$climateRows = foreach ($property in $climateTimeseriesResponse.data.PSObject.Properties) {
    $code = $property.Name
    $series = $property.Value
    $baseline = if ($climateBaselineByCode.ContainsKey($code)) { $climateBaselineByCode[$code] } else { $null }

    if ($null -eq $baseline) {
        $baselineValues = @()
        foreach ($year in 1995..2014) {
            $histKey = '{0}-07' -f $year
            if ($series.PSObject.Properties.Name -contains $histKey -and $null -ne $series.$histKey) {
                $baselineValues += [double]$series.$histKey
            }
        }
        if (@($baselineValues).Count -gt 0) {
            $baseline = ($baselineValues | Measure-Object -Average).Average
        }
    }

    foreach ($year in $years) {
        $seriesKey = '{0}-07' -f $year
        $tasAnnual = if ($series.PSObject.Properties.Name -contains $seriesKey -and $null -ne $series.$seriesKey) {
            [double]$series.$seriesKey
        } else {
            $null
        }

        [PSCustomObject][ordered]@{
            code = $code
            year = $year
            climate_tas_annual_cru = if ($null -ne $tasAnnual) { [Math]::Round($tasAnnual, 3) } else { $null }
            climate_tas_baseline_1995_2014 = if ($null -ne $baseline) { [Math]::Round($baseline, 3) } else { $null }
            climate_anomaly = if (($null -ne $tasAnnual) -and ($null -ne $baseline)) { [Math]::Round($tasAnnual - $baseline, 3) } else { $null }
            climate_source = 'World Bank CCKP CRU TS4.08 annual tas anomaly vs 1995-2014 climatology'
        }
    }
}

$climateRows |
    Sort-Object code, year |
    Export-Csv -LiteralPath $climateCsvPath -NoTypeInformation -Encoding UTF8

$manifest = [ordered]@{
    generated_at = (Get-Date).ToString('s')
    outputs = [ordered]@{
        who_json = $whoJsonPath
        who_csv = $whoCsvPath
        climate_timeseries_json = $climateTimeseriesJsonPath
        climate_baseline_json = $climateBaselineJsonPath
        climate_csv = $climateCsvPath
    }
    sources = [ordered]@{
        who_dimension_values = $whoUrl
        climate_timeseries = $climateTimeseriesUrl
        climate_baseline = $climateBaselineUrl
    }
    notes = @(
        'Indonesia was reset to South-East Asia for this study because WHO reassigned Indonesia to WPR in 2025, after the 2016-2024 analysis window.',
        'CCKP observed CRU TS4.08 annual data currently end in 2023, so 2024 climate anomaly rows remain blank.'
    )
}

$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

@(
    "WHO mapping: $whoCsvPath"
    "Climate anomaly: $climateCsvPath"
    "Manifest: $manifestPath"
) -join "`r`n"
