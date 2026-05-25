$ErrorActionPreference = 'Stop'

$root = 'C:\Users\adamk\Documents\New project\downloads\bmjgh_sources'
$csvRoot = Join-Path $root 'csv'
$auxRoot = Join-Path $root 'aux'
$analysisDir = Join-Path $root 'analysis'
$panelPath = Join-Path $analysisDir 'bmjgh_main_panel_2016_2024.csv'
$qcJsonPath = Join-Path $analysisDir 'bmjgh_main_panel_qc.json'
$qcMdPath = Join-Path $analysisDir 'bmjgh_main_panel_qc.md'

New-Item -ItemType Directory -Force -Path $analysisDir | Out-Null

$years = 2016..2024
$culture = [System.Globalization.CultureInfo]::InvariantCulture

function New-PanelKey {
    param(
        [Parameter(Mandatory = $true)][string]$Code,
        [Parameter(Mandatory = $true)][int]$Year
    )

    '{0}|{1}' -f $Code, $Year
}

function Convert-ToNumber {
    param([AllowNull()][string]$Value)

    if ($null -eq $Value) { return $null }

    $clean = $Value.ToString().Trim()
    if ([string]::IsNullOrWhiteSpace($clean)) { return $null }
    if ($clean -in @('..', '...', 'NA', 'N/A')) { return $null }

    $clean = $clean -replace [char]0xA0, ' '
    $clean = $clean -replace '\s', ''
    $clean = $clean -replace ',', ''

    $parsed = 0.0
    if ([double]::TryParse($clean, [System.Globalization.NumberStyles]::Float, $culture, [ref]$parsed)) {
        return $parsed
    }

    return $null
}

function Normalize-Name {
    param([AllowNull()][string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) { return '' }

    $normalized = $Value.ToLowerInvariant()
    $normalized = $normalized -replace '&', ' and '
    $normalized = $normalized.Normalize([Text.NormalizationForm]::FormD)
    $chars = foreach ($ch in $normalized.ToCharArray()) {
        if ([Globalization.CharUnicodeInfo]::GetUnicodeCategory($ch) -ne [Globalization.UnicodeCategory]::NonSpacingMark) {
            $ch
        }
    }
    $normalized = -join $chars
    $normalized = $normalized -replace '[^a-z0-9 ]', ' '
    $normalized = $normalized -replace '\s+', ' '
    $normalized.Trim()
}

function Test-IsMissingValue {
    param($Value)

    if ($null -eq $Value) { return $true }
    if ($Value -is [string]) { return [string]::IsNullOrWhiteSpace($Value) }
    return $false
}

function Read-WuenicLong {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$MetricName
    )

    $rows = Import-Csv $Path | Where-Object { $_.iso3 -match '^[A-Z]{3}$' }
    foreach ($row in $rows) {
        foreach ($year in $years) {
            [PSCustomObject]@{
                code = $row.iso3
                country_name = $row.country
                unicef_region = $row.unicef_region
                year = $year
                value = Convert-ToNumber $row.$year
                metric = $MetricName
            }
        }
    }
}

function Read-WorldBankIndicatorLong {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$MetricName
    )

    $rows = (Get-Content $Path | Select-Object -Skip 3) | ConvertFrom-Csv
    foreach ($row in $rows) {
        $code = $row.'Country Code'
        if ($code -notmatch '^[A-Z]{3}$') { continue }

        foreach ($year in $years) {
            [PSCustomObject]@{
                code = $code
                country_name = $row.'Country Name'
                year = $year
                value = Convert-ToNumber $row.$year
                metric = $MetricName
            }
        }
    }
}

function Read-WppLong {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$VariantName
    )

    $rows = (Get-Content $Path | Select-Object -Skip 17) | ConvertFrom-Csv
    $rows |
        Where-Object {
            $_.Type -eq 'Country/Area' -and
            $_.'ISO3 Alpha-code' -match '^[A-Z]{3}$' -and
            [int]$_.Year -in $years
        } |
        ForEach-Object {
            [PSCustomObject]@{
                code = $_.'ISO3 Alpha-code'
                country_name = $_.'Region, subregion, country or area *'
                year = [int]$_.Year
                variant = $VariantName
                population_total_wpp = if ($null -ne (Convert-ToNumber $_.'Total Population, as of 1 July (thousands)')) { (Convert-ToNumber $_.'Total Population, as of 1 July (thousands)') * 1000 } else { $null }
                birth_rate_crude_wpp = Convert-ToNumber $_.'Crude Birth Rate (births per 1,000 population)'
                births_wpp = if ($null -ne (Convert-ToNumber $_.'Births (thousands)')) { (Convert-ToNumber $_.'Births (thousands)') * 1000 } else { $null }
                surviving_infants_wpp = if ($null -ne (Convert-ToNumber $_.'Live Births Surviving to Age 1 (thousands)')) { (Convert-ToNumber $_.'Live Births Surviving to Age 1 (thousands)') * 1000 } else { $null }
            }
        }
}

function Read-HistoricalIncomeLong {
    param([Parameter(Mandatory = $true)][string]$Path)

    $rows = Import-Csv $Path
    $yearRow = $rows[5]
    $dataRows = $rows | Where-Object { $_.F1 -match '^[A-Z]{3}$' }

    $yearToColumn = @{}
    foreach ($property in $yearRow.PSObject.Properties) {
        if ($property.Value -match '^\d{4}$') {
            $yearToColumn[[int]$property.Value] = $property.Name
        }
    }

    $incomeLabels = @{
        'L' = 'Low income'
        'LM' = 'Lower middle income'
        'UM' = 'Upper middle income'
        'H' = 'High income'
    }

    foreach ($row in $dataRows) {
        foreach ($year in $years) {
            if (-not $yearToColumn.ContainsKey($year)) { continue }

            $columnName = $yearToColumn[$year]
            $incomeCode = $row.$columnName
            $incomeCode = if ($incomeCode) { $incomeCode.Trim() } else { $null }

            [PSCustomObject]@{
                code = $row.F1
                country_name = $row.'World Bank Analytical Classifications'
                year = $year
                income_group = if ($incomeLabels.ContainsKey($incomeCode)) { $incomeLabels[$incomeCode] } else { $null }
                income_group_code = if ($incomeLabels.ContainsKey($incomeCode)) { $incomeCode } else { $null }
            }
        }
    }
}

function Get-ConflictAliasMap {
    @{
        'antigua and barbuda' = 'ATG'
        'antigua barbuda' = 'ATG'
        'bolivia' = 'BOL'
        'bosnia herzegovina' = 'BIH'
        'brunei' = 'BRN'
        'cambodia kampuchea' = 'KHM'
        'cape verde' = 'CPV'
        'czech republic' = 'CZE'
        'dr congo zaire' = 'COD'
        'east timor' = 'TLS'
        'federated states of micronesia' = 'FSM'
        'iran' = 'IRN'
        'ivory coast' = 'CIV'
        'kingdom of eswatini swaziland' = 'SWZ'
        'laos' = 'LAO'
        'moldova' = 'MDA'
        'myanmar burma' = 'MMR'
        'netherlands' = 'NLD'
        'north korea' = 'PRK'
        'russia soviet union' = 'RUS'
        'samoa western samoa' = 'WSM'
        'serbia yugoslavia' = 'SRB'
        'south korea' = 'KOR'
        'syria' = 'SYR'
        'tanzania' = 'TZA'
        'turkey' = 'TUR'
        'united states of america' = 'USA'
        'venezuela' = 'VEN'
        'vietnam north vietnam' = 'VNM'
        'yemen north yemen' = 'YEM'
        'zimbabwe rhodesia' = 'ZWE'
        'madagascar malagasy' = 'MDG'
    }
}

function Read-UcdpConflictLong {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][hashtable]$CountryNameMap,
        [Parameter(Mandatory = $true)][hashtable]$AliasMap
    )

    $rows = Import-Csv $Path | Where-Object { [int]$_.year_cy -in $years }
    foreach ($row in $rows) {
        $normalized = Normalize-Name $row.country_cy
        $code = $null

        if ($CountryNameMap.ContainsKey($normalized)) {
            $code = $CountryNameMap[$normalized]
        }
        elseif ($AliasMap.ContainsKey($normalized)) {
            $code = $AliasMap[$normalized]
        }

        if (-not $code) { continue }

        $deaths = Convert-ToNumber $row.cumulative_total_deaths_in_orgvio_best_cy
        $sb = Convert-ToNumber $row.sb_exist_cy
        $ns = Convert-ToNumber $row.ns_exist_cy
        $os = Convert-ToNumber $row.os_exist_cy

        [PSCustomObject]@{
            code = $code
            year = [int]$row.year_cy
            conflict = if (($sb -eq 1) -or ($ns -eq 1) -or ($os -eq 1) -or ($deaths -gt 0)) { 1 } else { 0 }
            conflict_deaths_best = if ($null -ne $deaths) { $deaths } else { 0 }
        }
    }
}

function Get-FcsCodeMap {
    @{
        2016 = @(
            'AFG', 'BDI', 'CAF', 'TCD', 'COM', 'COD', 'CIV', 'ERI', 'GNB', 'HTI',
            'KIR', 'MDG', 'MHL', 'FSM', 'MMR', 'SOM', 'SSD', 'SDN', 'TLS', 'TUV',
            'BIH', 'IRQ', 'LBN', 'LBY', 'SYR', 'PSE', 'YEM', 'MLI', 'SLE', 'ZWE'
        )
        2017 = @(
            'AFG', 'BDI', 'CMR', 'CAF', 'TCD', 'COM', 'COD', 'CIV', 'ERI', 'GNB',
            'LBR', 'MHL', 'FSM', 'PNG', 'SLB', 'SSD', 'TGO', 'PSE', 'YEM', 'BIH',
            'MLI', 'SLE', 'ZWE'
        )
        2018 = @(
            'AFG', 'BDI', 'CMR', 'CAF', 'TCD', 'COM', 'COD', 'CIV', 'ERI', 'GNB',
            'XKX', 'LBR', 'MHL', 'FSM', 'PNG', 'STP', 'SLB', 'SSD', 'TGO', 'PSE',
            'YEM', 'MLI', 'MOZ', 'NER', 'NGA', 'SLE', 'ZWE'
        )
        2019 = @(
            'AFG', 'BDI', 'CAF', 'TCD', 'COM', 'COD', 'COG', 'CIV', 'DJI', 'ERI',
            'GMB', 'GNB', 'HTI', 'KIR', 'XKX', 'LBR', 'MLI', 'MHL', 'FSM', 'MOZ',
            'MMR', 'PNG', 'SLE', 'SLB', 'SOM', 'SSD', 'SDN', 'SYR', 'TGO', 'TUV',
            'YEM', 'PSE', 'ZWE', 'IRQ', 'LBN', 'LBY'
        )
        2020 = @(
            'AFG', 'CAF', 'LBY', 'SOM', 'SSD', 'SYR', 'YEM', 'BFA', 'BDI', 'CMR',
            'COD', 'IRQ', 'MLI', 'NER', 'NGA', 'SDN', 'TCD', 'COG', 'ERI', 'GMB',
            'GNB', 'HTI', 'XKX', 'LBN', 'LBR', 'MMR', 'PNG', 'VEN', 'ZWE', 'PSE',
            'COM', 'KIR', 'MHL', 'FSM', 'SLB', 'TLS', 'TUV', 'CIV', 'DJI', 'MOZ',
            'TGO'
        )
        2021 = @(
            'AFG', 'LBY', 'SOM', 'SYR', 'BFA', 'CMR', 'CAF', 'TCD', 'COD', 'IRQ',
            'MLI', 'MOZ', 'MMR', 'NER', 'NGA', 'SSD', 'YEM', 'BDI', 'COG', 'ERI',
            'GMB', 'GNB', 'HTI', 'XKX', 'LAO', 'LBN', 'LBR', 'PNG', 'SDN', 'VEN',
            'PSE', 'ZWE', 'COM', 'KIR', 'MHL', 'FSM', 'SLB', 'TLS', 'TUV'
        )
        2022 = @(
            'AFG', 'SOM', 'SYR', 'YEM', 'BFA', 'BDI', 'CMR', 'CAF', 'TCD', 'COD',
            'ETH', 'HTI', 'IRQ', 'LBY', 'MLI', 'MOZ', 'MMR', 'NER', 'NGA', 'SSD',
            'COG', 'ERI', 'GNB', 'XKX', 'LBN', 'PNG', 'SDN', 'VEN', 'PSE', 'ZWE',
            'ARM', 'AZE', 'COM', 'KIR', 'MHL', 'FSM', 'SLB', 'TUV'
        )
        2023 = @(
            'AFG', 'BFA', 'CMR', 'CAF', 'COD', 'ETH', 'IRQ', 'MLI', 'MOZ', 'MMR',
            'NER', 'NGA', 'SOM', 'SSD', 'SDN', 'SYR', 'UKR', 'YEM', 'BDI', 'TCD',
            'COM', 'COG', 'ERI', 'GNB', 'HTI', 'XKX', 'LBN', 'LBY', 'MHL', 'FSM',
            'PNG', 'SLB', 'TLS', 'TUV', 'VEN', 'PSE', 'ZWE'
        )
        2024 = @(
            'AFG', 'BFA', 'CMR', 'CAF', 'COD', 'ETH', 'IRQ', 'MLI', 'MOZ', 'MMR',
            'NER', 'NGA', 'SOM', 'SSD', 'SDN', 'SYR', 'UKR', 'YEM', 'BDI', 'TCD',
            'COM', 'COG', 'ERI', 'GNB', 'HTI', 'KIR', 'XKX', 'LBN', 'LBY', 'MHL',
            'FSM', 'PNG', 'STP', 'SLB', 'TLS', 'TUV', 'VEN', 'PSE', 'ZWE'
        )
    }
}

$dtp1Path = Join-Path $csvRoot 'wuenic2024rev_web-update\DTP1.csv'
$dtp3Path = Join-Path $csvRoot 'wuenic2024rev_web-update\DTP3.csv'
$mcv1Path = Join-Path $csvRoot 'wuenic2024rev_web-update\MCV1.csv'
$wbPopPath = Join-Path $csvRoot 'world_bank_SP_POP_TOTL\Data.csv'
$wbBirthPath = Join-Path $csvRoot 'world_bank_SP_DYN_CBRT_IN\Data.csv'
$wbHealthPath = Join-Path $csvRoot 'world_bank_SH_XPD_CHEX_GD_ZS\Data.csv'
$wbIncomeCurrentPath = Join-Path $csvRoot 'world_bank_current_income_classification\List_of_economies.csv'
$wbIncomeHistoricalPath = Join-Path $csvRoot 'world_bank_historical_income_classification\Country_Analytical_History.csv'
$wppEstimatesPath = Join-Path $csvRoot 'WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT\Estimates.csv'
$wppMediumPath = Join-Path $csvRoot 'WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT\Medium_variant.csv'
$ucdpPath = Join-Path $csvRoot 'ucdp_organizedviolencecy_251_csv\organizedviolencecy_v25_1.csv'
$whoRegionPath = Join-Path $auxRoot 'who_country_region_mapping.csv'
$climateAnomalyPath = Join-Path $auxRoot 'cckp_cru_tas_anomaly_2016_2024.csv'

foreach ($requiredPath in @($whoRegionPath, $climateAnomalyPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Missing required auxiliary file: $requiredPath. Run prepare_bmjgh_aux_sources.ps1 first."
    }
}

$dtp1Long = @(Read-WuenicLong -Path $dtp1Path -MetricName 'dtp1')
$dtp3Long = @(Read-WuenicLong -Path $dtp3Path -MetricName 'dtp3')
$mcv1Long = @(Read-WuenicLong -Path $mcv1Path -MetricName 'mcv1')

$panel = @{}
foreach ($row in $dtp3Long) {
    $key = New-PanelKey -Code $row.code -Year $row.year
    $panel[$key] = [PSCustomObject][ordered]@{
        code = $row.code
        country_name = $row.country_name
        year = $row.year
        unicef_region = $row.unicef_region
        who_region = $null
        wb_region = $null
        dtp1 = $null
        dtp3 = $row.value
        mcv1 = $null
        population_total = $null
        population_total_source = $null
        birth_rate_crude = $null
        birth_rate_crude_source = $null
        live_births_approx = $null
        births_wpp = $null
        surviving_infants = $null
        health_exp_gdp = $null
        income_group = $null
        income_group_source = $null
        fragile_status = $null
        fcs = $null
        conflict = 0
        conflict_deaths_best = 0
        climate_anomaly = $null
    }
}

foreach ($row in $dtp1Long) {
    $key = New-PanelKey -Code $row.code -Year $row.year
    if ($panel.ContainsKey($key)) { $panel[$key].dtp1 = $row.value }
}

foreach ($row in $mcv1Long) {
    $key = New-PanelKey -Code $row.code -Year $row.year
    if ($panel.ContainsKey($key)) { $panel[$key].mcv1 = $row.value }
}

$baseCodes = $panel.Values.code | Sort-Object -Unique
$baseCodeLookup = @{}
foreach ($code in $baseCodes) { $baseCodeLookup[$code] = $true }

foreach ($row in (Read-WorldBankIndicatorLong -Path $wbPopPath -MetricName 'population_total')) {
    if (-not $baseCodeLookup.ContainsKey($row.code)) { continue }
    $key = New-PanelKey -Code $row.code -Year $row.year
    if ($panel.ContainsKey($key) -and $null -ne $row.value) {
        $panel[$key].population_total = $row.value
        $panel[$key].population_total_source = 'World Bank'
    }
}

foreach ($row in (Read-WorldBankIndicatorLong -Path $wbBirthPath -MetricName 'birth_rate_crude')) {
    if (-not $baseCodeLookup.ContainsKey($row.code)) { continue }
    $key = New-PanelKey -Code $row.code -Year $row.year
    if ($panel.ContainsKey($key) -and $null -ne $row.value) {
        $panel[$key].birth_rate_crude = $row.value
        $panel[$key].birth_rate_crude_source = 'World Bank'
    }
}

foreach ($row in (Read-WorldBankIndicatorLong -Path $wbHealthPath -MetricName 'health_exp_gdp')) {
    if (-not $baseCodeLookup.ContainsKey($row.code)) { continue }
    $key = New-PanelKey -Code $row.code -Year $row.year
    if ($panel.ContainsKey($key)) {
        $panel[$key].health_exp_gdp = $row.value
    }
}

$wppRows = @()
$wppRows += Read-WppLong -Path $wppEstimatesPath -VariantName 'Estimates'
$wppRows += Read-WppLong -Path $wppMediumPath -VariantName 'Medium'

foreach ($row in $wppRows) {
    if (-not $baseCodeLookup.ContainsKey($row.code)) { continue }
    $key = New-PanelKey -Code $row.code -Year $row.year
    if (-not $panel.ContainsKey($key)) { continue }

    if ($null -eq $panel[$key].population_total -and $null -ne $row.population_total_wpp) {
        $panel[$key].population_total = $row.population_total_wpp
        $panel[$key].population_total_source = 'WPP'
    }

    if ($null -eq $panel[$key].birth_rate_crude -and $null -ne $row.birth_rate_crude_wpp) {
        $panel[$key].birth_rate_crude = $row.birth_rate_crude_wpp
        $panel[$key].birth_rate_crude_source = 'WPP'
    }

    if ($null -ne $row.births_wpp) {
        $panel[$key].births_wpp = $row.births_wpp
    }

    if ($null -ne $row.surviving_infants_wpp) {
        $panel[$key].surviving_infants = $row.surviving_infants_wpp
    }
}

$currentIncomeRows = Import-Csv $wbIncomeCurrentPath
$currentIncomeByCode = @{}
foreach ($row in $currentIncomeRows) {
    $currentIncomeByCode[$row.Code] = $row
}

$whoRegionRows = Import-Csv $whoRegionPath
$whoRegionByCode = @{}
foreach ($row in $whoRegionRows) {
    $whoRegionByCode[$row.code] = $row
}

foreach ($row in Read-HistoricalIncomeLong -Path $wbIncomeHistoricalPath) {
    if (-not $baseCodeLookup.ContainsKey($row.code)) { continue }
    $key = New-PanelKey -Code $row.code -Year $row.year
    if ($panel.ContainsKey($key) -and $null -ne $row.income_group) {
        $panel[$key].income_group = $row.income_group
        $panel[$key].income_group_source = 'World Bank historical'
    }
}

foreach ($item in $panel.Values) {
    if ($currentIncomeByCode.ContainsKey($item.code)) {
        $currentRow = $currentIncomeByCode[$item.code]
        $item.wb_region = $currentRow.Region
        if ($null -eq $item.income_group -and -not [string]::IsNullOrWhiteSpace($currentRow.'Income group')) {
            $item.income_group = $currentRow.'Income group'
            $item.income_group_source = 'World Bank current'
        }
    }

    if ($whoRegionByCode.ContainsKey($item.code)) {
        $item.who_region = $whoRegionByCode[$item.code].who_region
    }
}

$countryNameMap = @{}
foreach ($item in ($panel.Values | Sort-Object code -Unique)) {
    $normalized = Normalize-Name $item.country_name
    if (-not [string]::IsNullOrWhiteSpace($normalized)) {
        $countryNameMap[$normalized] = $item.code
    }
}

$aliasMap = Get-ConflictAliasMap
foreach ($row in (Read-UcdpConflictLong -Path $ucdpPath -CountryNameMap $countryNameMap -AliasMap $aliasMap)) {
    $key = New-PanelKey -Code $row.code -Year $row.year
    if ($panel.ContainsKey($key)) {
        $panel[$key].conflict = $row.conflict
        $panel[$key].conflict_deaths_best = $row.conflict_deaths_best
    }
}

$fcsLookup = @{}
foreach ($entry in (Get-FcsCodeMap).GetEnumerator()) {
    foreach ($code in $entry.Value) {
        $fcsLookup[(New-PanelKey -Code $code -Year $entry.Key)] = $true
    }
}

foreach ($item in $panel.Values) {
    $item.fcs = if ($fcsLookup.ContainsKey((New-PanelKey -Code $item.code -Year $item.year))) { 1 } else { 0 }
    $item.fragile_status = if ($item.fcs -eq 1) { 'FCS' } else { 'Non-FCS' }
}

foreach ($row in (Import-Csv $climateAnomalyPath)) {
    if (-not $baseCodeLookup.ContainsKey($row.code)) { continue }
    $key = New-PanelKey -Code $row.code -Year ([int]$row.year)
    if ($panel.ContainsKey($key)) {
        $panel[$key].climate_anomaly = Convert-ToNumber $row.climate_anomaly
    }
}

foreach ($item in $panel.Values) {
    if (($null -ne $item.population_total) -and ($null -ne $item.birth_rate_crude)) {
        $item.live_births_approx = $item.population_total * $item.birth_rate_crude / 1000.0
    }
}

$panelRows = $panel.Values | Sort-Object code, year
$panelRows | Export-Csv -LiteralPath $panelPath -NoTypeInformation -Encoding UTF8

$dtp3SourceRows = Import-Csv $dtp3Path
$mcv1SourceRows = Import-Csv $mcv1Path
$dtp1SourceRows = Import-Csv $dtp1Path

$duplicateRows = $panelRows | Group-Object code, year | Where-Object { $_.Count -gt 1 }
$countryYearCounts = $panelRows | Group-Object code | ForEach-Object {
    [PSCustomObject]@{
        code = $_.Name
        year_count = $_.Count
    }
}
$incompleteCountries = $countryYearCounts | Where-Object { $_.year_count -ne 9 }

$rangeCheck = foreach ($metric in @('dtp1', 'dtp3', 'mcv1')) {
    $violations = $panelRows | Where-Object {
        $value = $_.$metric
        $null -ne $value -and (($value -lt 0) -or ($value -gt 100))
    }
    [PSCustomObject]@{
        metric = $metric
        violation_count = @($violations).Count
    }
}

$covariateMissing = foreach ($field in @(
    'population_total',
    'birth_rate_crude',
    'live_births_approx',
    'surviving_infants',
    'health_exp_gdp',
    'income_group',
    'wb_region',
    'fragile_status',
    'conflict',
    'fcs',
    'climate_anomaly',
    'who_region'
)) {
    [PSCustomObject]@{
        field = $field
        missing_count = @($panelRows | Where-Object { Test-IsMissingValue $_.$field }).Count
    }
}

$qc = [ordered]@{
    assumptions = @(
        'No meASEL.csv was found in the workspace; MCV1 was taken from WUENIC MCV1.csv.',
        'WHO region was mapped from the official WHO GHO country dimension values; Indonesia was reset to South-East Asia because the panel ends in 2024 and WHO reassigned Indonesia to WPR in 2025.',
        'FCS was mapped from World Bank FY16-FY24 fragile-situations / FCS lists by assigning each fiscal-year label to the matching calendar year in the panel.',
        'climate_anomaly uses official World Bank CCKP CRU TS4.08 annual country temperature relative to the 1995-2014 climatology baseline; the current CCKP observed series ends in 2023, so climate_anomaly is blank for 2024.',
        'World Bank crude birth rate is blank for 2024; WPP 2024 medium-variant values were used as fallback where needed.'
    )
    outputs = [ordered]@{
        panel_csv = $panelPath
        qc_json = $qcJsonPath
        qc_md = $qcMdPath
    }
    panel_shape = [ordered]@{
        row_count = @($panelRows).Count
        country_count = @($panelRows | Group-Object code).Count
        year_range = '2016-2024'
    }
    check_1_code_year_unique = [ordered]@{
        duplicate_count = @($duplicateRows).Count
    }
    check_2_outcome_ranges = $rangeCheck
    check_3_balanced_2016_2024 = [ordered]@{
        countries_with_9_years = @($countryYearCounts | Where-Object { $_.year_count -eq 9 }).Count
        countries_not_balanced = @($incompleteCountries).Count
        incomplete_codes = [object[]]@($incompleteCountries.code | Where-Object { $_ })
    }
    check_4_region_missingness = [ordered]@{
        unicef_region_missing = @($panelRows | Where-Object { [string]::IsNullOrWhiteSpace($_.unicef_region) }).Count
        who_region_missing = @($panelRows | Where-Object { [string]::IsNullOrWhiteSpace($_.who_region) }).Count
        wb_region_missing = @($panelRows | Where-Object { [string]::IsNullOrWhiteSpace($_.wb_region) }).Count
    }
    check_5_covariate_missingness = $covariateMissing
    check_6_mcv1_country_filter = [ordered]@{
        source_row_count = @($mcv1SourceRows).Count
        source_non_country_rows = @($mcv1SourceRows | Where-Object { $_.iso3 -notmatch '^[A-Z]{3}$' }).Count
        retained_country_rows = @($mcv1SourceRows | Where-Object { $_.iso3 -match '^[A-Z]{3}$' }).Count
    }
    outcome_non_missing = [ordered]@{
        dtp1_non_missing = @($panelRows | Where-Object { $null -ne $_.dtp1 }).Count
        dtp3_non_missing = @($panelRows | Where-Object { $null -ne $_.dtp3 }).Count
        mcv1_non_missing = @($panelRows | Where-Object { $null -ne $_.mcv1 }).Count
    }
}

$qc | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $qcJsonPath -Encoding UTF8

$qcMd = @(
    '# BMJGH Main Panel QC'
    ''
    '## Outputs'
    "- Panel: $panelPath"
    "- QC JSON: $qcJsonPath"
    ''
    '## Assumptions'
)

foreach ($line in $qc.assumptions) {
    $qcMd += "- $line"
}

$qcMd += ''
$qcMd += '## Panel Shape'
$qcMd += "- Rows: $($qc.panel_shape.row_count)"
$qcMd += "- Countries: $($qc.panel_shape.country_count)"
$qcMd += "- Years: $($qc.panel_shape.year_range)"
$qcMd += ''
$qcMd += '## Checks'
$qcMd += "- code-year duplicates: $($qc.check_1_code_year_unique.duplicate_count)"

foreach ($item in $qc.check_2_outcome_ranges) {
    $qcMd += "- $($item.metric) range violations: $($item.violation_count)"
}

$qcMd += "- balanced countries with 9 years: $($qc.check_3_balanced_2016_2024.countries_with_9_years)"
$qcMd += "- unicef_region missing: $($qc.check_4_region_missingness.unicef_region_missing)"
$qcMd += "- who_region missing: $($qc.check_4_region_missingness.who_region_missing)"
$qcMd += "- wb_region missing: $($qc.check_4_region_missingness.wb_region_missing)"

foreach ($item in $qc.check_5_covariate_missingness) {
    $qcMd += "- $($item.field) missing: $($item.missing_count)"
}

$qcMd += "- MCV1 source non-country rows: $($qc.check_6_mcv1_country_filter.source_non_country_rows)"

$qcMd -join "`r`n" | Set-Content -LiteralPath $qcMdPath -Encoding UTF8

$panelRows |
    Select-Object -First 10 |
    Format-Table -AutoSize code, country_name, year, unicef_region, dtp1, dtp3, mcv1, population_total, birth_rate_crude, live_births_approx, health_exp_gdp, income_group, conflict
