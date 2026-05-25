$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$root = 'C:\Users\adamk\Documents\New project\downloads\bmjgh_sources'
$rawDir = Join-Path $root 'raw'
$extractDir = Join-Path $root 'extracted'
$manifestPath = Join-Path $root 'download_manifest.json'

foreach ($dir in @($root, $rawDir, $extractDir)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

$userAgent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0 Safari/537.36'

$sources = @(
    @{
        key = 'wuenic_excel'
        title = 'WUENIC 2024 revision (all-country Excel)'
        kind = 'direct_file'
        expectedExt = '.xlsx'
        url = 'https://data.unicef.org/wp-content/uploads/2024/07/wuenic2024rev_web-update.xlsx'
        baseName = 'wuenic2024rev_web-update'
        note = 'Main immunization coverage workbook.'
        referer = 'https://data.unicef.org/resources/dataset/immunization/'
    },
    @{
        key = 'wuenic_notes_pdf'
        title = 'WHO/UNICEF WUENIC notes for all countries (2024 revision)'
        kind = 'direct_file'
        expectedExt = '.pdf'
        url = 'https://www.who.int/docs/default-source/immunization/immunization-coverage/wuenic_notes.pdf?sfvrsn=88ff590d_6'
        baseName = 'WUENIC_notes_for_AllCountries_2024rev'
        note = 'Methods and weighting notes.'
        referer = 'https://www.who.int/teams/immunization-vaccines-and-biologicals/immunization-analysis-and-insights/global-monitoring/immunization-coverage/who-unicef-estimates-of-national-immunization-coverage'
    },
    @{
        key = 'who_all_vaccination_coverage_export'
        title = 'WHO Immunization Data portal all vaccination coverage export'
        kind = 'direct_file'
        expectedExt = '.xlsx'
        url = 'https://srhdpeuwpubsa-geecgzbpd5h0fueu.z01.azurefd.net/whdh/WIISE/export/coverage-data.xlsx'
        baseName = 'who_immunization_all_vaccination_coverage_data'
        note = 'WHO Immunization Data portal export for all vaccination coverage data.'
        referer = 'https://immunizationdata.who.int/listing.html'
    },
    @{
        key = 'wpp_download_center'
        title = 'World Population Prospects 2024 download center'
        kind = 'entry_page'
        expectedExt = '.html'
        url = 'https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=Most%20used'
        baseName = 'wpp_download_center'
        note = 'Official entry page for WPP downloads.'
    },
    @{
        key = 'wpp_demographic_indicators_compact'
        title = 'World Population Prospects 2024 demographic indicators (compact)'
        kind = 'direct_file'
        expectedExt = '.xlsx'
        url = 'https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/EXCEL_FILES/1_General/WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx'
        baseName = 'WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT'
        note = 'Most-used WPP demographic indicators workbook.'
    },
    @{
        key = 'wpp_methodology_pdf'
        title = 'World Population Prospects 2024 methodology'
        kind = 'direct_file'
        expectedExt = '.pdf'
        url = 'https://population.un.org/wpp/assets/Files/WPP2024_Methodology.pdf'
        baseName = 'WPP2024_Methodology'
        note = 'Official WPP methodology PDF.'
    },
    @{
        key = 'wpp_data_sources_pdf'
        title = 'World Population Prospects 2024 data sources'
        kind = 'direct_file'
        expectedExt = '.pdf'
        url = 'https://population.un.org/wpp/assets/Excel%20Files/4_Metadata/WPP2024_Data_Sources.pdf'
        baseName = 'WPP2024_Data_Sources'
        note = 'Official WPP data sources PDF.'
    },
    @{
        key = 'world_bank_population_total'
        title = 'World Bank Population, total (SP.POP.TOTL)'
        kind = 'direct_file'
        expectedExt = '.zip'
        url = 'https://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=excel'
        baseName = 'world_bank_SP_POP_TOTL'
        note = 'Indicator export, typically a ZIP package with Excel data and metadata.'
    },
    @{
        key = 'world_bank_birth_rate_crude'
        title = 'World Bank Birth rate, crude (SP.DYN.CBRT.IN)'
        kind = 'direct_file'
        expectedExt = '.zip'
        url = 'https://api.worldbank.org/v2/en/indicator/SP.DYN.CBRT.IN?downloadformat=excel'
        baseName = 'world_bank_SP_DYN_CBRT_IN'
        note = 'Indicator export, typically a ZIP package with Excel data and metadata.'
    },
    @{
        key = 'world_bank_health_expenditure'
        title = 'Current health expenditure (% of GDP) (SH.XPD.CHEX.GD.ZS)'
        kind = 'direct_file'
        expectedExt = '.zip'
        url = 'https://api.worldbank.org/v2/en/indicator/SH.XPD.CHEX.GD.ZS?downloadformat=excel'
        baseName = 'world_bank_SH_XPD_CHEX_GD_ZS'
        note = 'Indicator export, typically a ZIP package with Excel data and metadata.'
    },
    @{
        key = 'world_bank_income_historical'
        title = 'World Bank historical income classification'
        kind = 'direct_file'
        expectedExt = '.xlsx'
        url = 'https://ddh-openapi.worldbank.org/resources/DR0095334/download'
        baseName = 'world_bank_historical_income_classification'
        note = 'Historical World Bank income classification workbook.'
    },
    @{
        key = 'world_bank_income_current'
        title = 'World Bank current income classification'
        kind = 'direct_file'
        expectedExt = '.xlsx'
        url = 'https://ddh-openapi.worldbank.org/resources/DR0095333/download'
        baseName = 'world_bank_current_income_classification'
        note = 'Current World Bank income classification workbook.'
    },
    @{
        key = 'world_bank_fcs_page'
        title = 'World Bank FCS classification page'
        kind = 'entry_page'
        expectedExt = '.html'
        url = 'https://www.worldbank.org/en/topic/fragilityconflictviolence/brief/classification-of-fragile-and-conflict-affected-situations'
        baseName = 'world_bank_fcs_classification_page'
        note = 'Official entry page for FCS classification.'
    },
    @{
        key = 'world_bank_fcs_fy26_pdf'
        title = 'World Bank FCS FY26 list'
        kind = 'direct_file'
        expectedExt = '.pdf'
        url = 'https://thedocs.worldbank.org/en/doc/5c7e4e268baaafa6ef38d924be9279be-0090082025/original/FCSListFY26.pdf'
        baseName = 'FCSListFY26'
        note = 'FY26 fragile and conflict-affected situations list.'
    },
    @{
        key = 'world_bank_fcs_historical_pdf'
        title = 'World Bank FCS historical FY06-FY25 note'
        kind = 'direct_file'
        expectedExt = '.pdf'
        url = 'https://thedocs.worldbank.org/en/doc/373511582764863285-0090022020/original/FCSHistorialnote.pdf'
        baseName = 'FCSHistorialnote_FY06_FY25'
        note = 'Historical note and lists for FCS classifications.'
    },
    @{
        key = 'ucdp_organized_violence'
        title = 'UCDP Country-Year Dataset on Organized Violence within Country Borders v25.1'
        kind = 'direct_file'
        expectedExt = '.zip'
        url = 'https://ucdp.uu.se/downloads/organizedviolencecy/organizedviolencecy-251-csv.zip'
        baseName = 'ucdp_organizedviolencecy_251_csv'
        note = 'Country-year organized violence dataset.'
    },
    @{
        key = 'ucdp_battle_related_deaths'
        title = 'UCDP Battle-Related Deaths Dataset v25.1 (dyadic)'
        kind = 'direct_file'
        expectedExt = '.zip'
        url = 'https://ucdp.uu.se/downloads/brd/ucdp-brd-dyadic-251-csv.zip'
        baseName = 'ucdp_brd_dyadic_251_csv'
        note = 'Dyadic battle-related deaths dataset.'
    },
    @{
        key = 'cckp_download_page'
        title = 'World Bank Climate Change Knowledge Portal download page'
        kind = 'entry_page'
        expectedExt = '.html'
        url = 'https://climateknowledgeportal.worldbank.org/download-data'
        baseName = 'cckp_download_data'
        note = 'Official entry page for climate downloads and API structure.'
    },
    @{
        key = 'cckp_metadata_page'
        title = 'World Bank Climate Change Knowledge Portal metadata'
        kind = 'entry_page'
        expectedExt = '.html'
        url = 'https://climateknowledgeportal.worldbank.org/metadata'
        baseName = 'cckp_metadata'
        note = 'Official metadata page with available variables and periods.'
    },
    @{
        key = 'cckp_spatial_codes_json'
        title = 'CCKP spatial unit names and codes JSON'
        kind = 'direct_file'
        expectedExt = '.json'
        url = 'https://climateknowledgeportal.worldbank.org/themes/custom/cckpmodern/data/geonames.json'
        baseName = 'cckp_spatial_unit_codes'
        note = 'Spatial unit names and codes JSON linked from the official CCKP download page.'
    },
    @{
        key = 'cckp_spatial_codes_xlsx'
        title = 'CCKP spatial unit names and codes Excel'
        kind = 'direct_file'
        expectedExt = '.xlsx'
        url = 'https://climateknowledgeportal.worldbank.org/themes/custom/cckpmodern/data/geonames.xlsx'
        baseName = 'cckp_spatial_unit_codes'
        note = 'Spatial unit names and codes Excel linked from the official CCKP download page.'
    },
    @{
        key = 'cckp_eez_tidegauges_json'
        title = 'CCKP EEZ and tide gauge codes JSON'
        kind = 'direct_file'
        expectedExt = '.json'
        url = 'https://climateknowledgeportal.worldbank.org/themes/custom/cckpmodern/data/EEZs_Tidegauges.json'
        baseName = 'cckp_EEZs_Tidegauges'
        note = 'EEZ and tide gauge code list linked from the official CCKP download page.'
    },
    @{
        key = 'cckp_eez_tidegauges_xlsx'
        title = 'CCKP EEZ and tide gauge codes Excel'
        kind = 'direct_file'
        expectedExt = '.xlsx'
        url = 'https://climateknowledgeportal.worldbank.org/themes/custom/cckpmodern/data/EEZs_Tidegauges.xlsx'
        baseName = 'cckp_EEZs_Tidegauges'
        note = 'EEZ and tide gauge code list linked from the official CCKP download page.'
    },
    @{
        key = 'era5_cckp_readme'
        title = 'ERA5 data collections on AWS (CCKP README)'
        kind = 'entry_page'
        expectedExt = '.html'
        url = 'https://worldbank.github.io/climateknowledgeportal/README.html'
        baseName = 'worldbank_climateknowledgeportal_README'
        note = 'Official README describing public AWS bucket collections.'
    }
)

function Get-DetectedExtension {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Fallback
    )

    $bytes = [System.IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -ge 4) {
        if ($bytes[0] -eq 0xD0 -and $bytes[1] -eq 0xCF -and $bytes[2] -eq 0x11 -and $bytes[3] -eq 0xE0) { return '.xls' }
        if ($bytes[0] -eq 0x50 -and $bytes[1] -eq 0x4B -and $Fallback -eq '.xlsx') { return '.xlsx' }
        if ($bytes[0] -eq 0x50 -and $bytes[1] -eq 0x4B) { return '.zip' }
        if ($bytes[0] -eq 0x25 -and $bytes[1] -eq 0x50 -and $bytes[2] -eq 0x44 -and $bytes[3] -eq 0x46) { return '.pdf' }
    }

    $textHeader = ''
    if ($bytes.Length -gt 0) {
        $sampleLength = [Math]::Min(256, $bytes.Length)
        $textHeader = [System.Text.Encoding]::UTF8.GetString($bytes, 0, $sampleLength)
    }

    if ($textHeader -match '^\s*<!DOCTYPE html|^\s*<html|^\s*Sorry, you need to enable JavaScript') { return '.html' }
    if ($textHeader -match '^\s*[\{\[]') { return '.json' }

    return $Fallback
}

function Download-Source {
    param([hashtable]$Source)

    $tempPath = Join-Path $rawDir ($Source.baseName + '.download')
    if (Test-Path $tempPath) {
        Remove-Item -LiteralPath $tempPath -Force
    }

    $result = [ordered]@{
        key = $Source.key
        title = $Source.title
        url = $Source.url
        kind = $Source.kind
        expectedExt = $Source.expectedExt
        localPath = $null
        extractedTo = $null
        sizeBytes = $null
        status = 'pending'
        note = $Source.note
        error = $null
        downloadedAt = (Get-Date).ToString('s')
    }

    try {
        $headers = @{ 'User-Agent' = $userAgent }
        if ($Source.ContainsKey('referer')) {
            $headers['Referer'] = $Source.referer
        }

        try {
            Invoke-WebRequest -Uri $Source.url -OutFile $tempPath -Headers $headers -MaximumRedirection 10
        }
        catch {
            $curlArgs = @(
                '--fail',
                '--location',
                '--retry', '2',
                '--user-agent', $userAgent,
                '--output', $tempPath
            )

            if ($Source.ContainsKey('referer')) {
                $curlArgs += @('--referer', $Source.referer)
            }

            $curlArgs += $Source.url
            & curl.exe @curlArgs
            if ($LASTEXITCODE -ne 0) {
                throw
            }
        }

        $detectedExt = Get-DetectedExtension -Path $tempPath -Fallback $Source.expectedExt
        $finalPath = Join-Path $rawDir ($Source.baseName + $detectedExt)

        Get-ChildItem -LiteralPath $rawDir -Filter ($Source.baseName + '.*') -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -ne $tempPath } |
            ForEach-Object {
                Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
            }

        if (Test-Path $finalPath) {
            Remove-Item -LiteralPath $finalPath -Force
        }
        Move-Item -LiteralPath $tempPath -Destination $finalPath

        $result.localPath = $finalPath
        $result.sizeBytes = (Get-Item -LiteralPath $finalPath).Length
        $result.status = 'downloaded'

        if ($detectedExt -eq '.zip' -and $Source.expectedExt -eq '.zip') {
            $targetExtractDir = Join-Path $extractDir $Source.baseName
            if (Test-Path $targetExtractDir) {
                Remove-Item -LiteralPath $targetExtractDir -Recurse -Force
            }
            New-Item -ItemType Directory -Force -Path $targetExtractDir | Out-Null
            Expand-Archive -LiteralPath $finalPath -DestinationPath $targetExtractDir -Force
            $result.extractedTo = $targetExtractDir
            $result.status = 'downloaded_and_extracted'
        }
        else {
            $staleExtractDir = Join-Path $extractDir $Source.baseName
            if (Test-Path $staleExtractDir) {
                Remove-Item -LiteralPath $staleExtractDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }
    catch {
        if (Test-Path $tempPath) {
            Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
        }
        $result.status = 'failed'
        $result.error = $_.Exception.Message
    }

    [PSCustomObject]$result
}

$results = foreach ($source in $sources) {
    Download-Source -Source $source
}

$results | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
$results | Sort-Object status, title | Format-Table -AutoSize key, status, sizeBytes, localPath, extractedTo
