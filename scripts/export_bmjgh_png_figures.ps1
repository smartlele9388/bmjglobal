$ErrorActionPreference = 'Stop'

$analysisDir = 'C:\Users\adamk\Documents\New project\downloads\bmjgh_sources\analysis\recovery_analysis'
$figDir = Join-Path $analysisDir 'figures_png'
$tableDir = Join-Path $analysisDir 'tables_main_text'

New-Item -ItemType Directory -Force -Path $figDir | Out-Null
New-Item -ItemType Directory -Force -Path $tableDir | Out-Null

Add-Type -AssemblyName System.Drawing

function Convert-ToNumber {
    param($Value)

    if ($null -eq $Value) { return $null }
    $text = $Value.ToString().Trim()
    if ([string]::IsNullOrWhiteSpace($text)) { return $null }
    $parsed = 0.0
    if ([double]::TryParse($text, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$parsed)) {
        return $parsed
    }

    return $null
}

function New-Font {
    param(
        [float]$Size,
        [System.Drawing.FontStyle]$Style = [System.Drawing.FontStyle]::Regular
    )

    return New-Object System.Drawing.Font('Segoe UI', $Size, $Style)
}

function Draw-PanelLabel {
    param(
        $Graphics,
        [string]$Text,
        [int]$X,
        [int]$Y
    )

    $font = New-Font -Size 16 -Style Bold
    $brush = [System.Drawing.Brushes]::Black
    $Graphics.DrawString($Text, $font, $brush, $X, $Y)
}

function Draw-LinePanel {
    param(
        $Graphics,
        [System.Drawing.Rectangle]$Rect,
        [string]$Title,
        [array]$SeriesConfigs,
        [int[]]$Years,
        [double]$YMin,
        [double]$YMax,
        [string]$YLabel
    )

    $titleFont = New-Font -Size 14 -Style Bold
    $labelFont = New-Font -Size 10
    $axisPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(60,60,60), 1.5)
    $gridPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(225,225,225), 1)
    $gridPen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash

    $plot = [System.Drawing.Rectangle]::new($Rect.X + 70, $Rect.Y + 45, $Rect.Width - 110, $Rect.Height - 100)
    $Graphics.DrawString($Title, $titleFont, [System.Drawing.Brushes]::Black, $Rect.X + 8, $Rect.Y + 8)

    $Graphics.DrawRectangle($axisPen, $plot)
    $tickCount = 5
    for ($i = 0; $i -le $tickCount; $i++) {
        $yValue = $YMin + ($YMax - $YMin) * $i / $tickCount
        $py = $plot.Bottom - ($plot.Height * $i / $tickCount)
        $Graphics.DrawLine($gridPen, $plot.X, $py, $plot.Right, $py)
        $label = [Math]::Round($yValue, 0).ToString()
        $Graphics.DrawString($label, $labelFont, [System.Drawing.Brushes]::DimGray, $Rect.X + 10, $py - 8)
    }

    $nYears = $Years.Count
    for ($i = 0; $i -lt $nYears; $i++) {
        $x = $plot.X + ($plot.Width * $i / ($nYears - 1))
        $Graphics.DrawLine($axisPen, $x, $plot.Bottom, $x, $plot.Bottom + 4)
        $Graphics.DrawString($Years[$i].ToString(), $labelFont, [System.Drawing.Brushes]::DimGray, $x - 14, $plot.Bottom + 8)
    }

    $sf = New-Object System.Drawing.StringFormat
    $sf.Alignment = [System.Drawing.StringAlignment]::Center
    $sf.LineAlignment = [System.Drawing.StringAlignment]::Center
    $Graphics.TranslateTransform($Rect.X + 18, $Rect.Y + ($Rect.Height / 2))
    $Graphics.RotateTransform(-90)
    $Graphics.DrawString($YLabel, $labelFont, [System.Drawing.Brushes]::DimGray, 0, 0, $sf)
    $Graphics.ResetTransform()

    foreach ($series in $SeriesConfigs) {
        $pen = New-Object System.Drawing.Pen($series.Color, 3)
        $brush = New-Object System.Drawing.SolidBrush($series.Color)
        $prevPoint = $null
        foreach ($year in $Years) {
            $row = $series.Data | Where-Object { [int]$_.year -eq $year } | Select-Object -First 1
            $value = Convert-ToNumber $row.mean
            $x = $plot.X + ($plot.Width * ([array]::IndexOf($Years, $year)) / ($nYears - 1))
            $y = $plot.Bottom - (($value - $YMin) / ($YMax - $YMin) * $plot.Height)
            if ($prevPoint) {
                $Graphics.DrawLine($pen, $prevPoint.X, $prevPoint.Y, $x, $y)
            }
            $Graphics.FillEllipse($brush, $x - 4, $y - 4, 8, 8)
            $prevPoint = [PSCustomObject]@{ X = $x; Y = $y }
        }
    }

    $legendX = $plot.Right - 160
    $legendY = $Rect.Y + 20
    $legendFont = New-Font -Size 9
    $idx = 0
    foreach ($series in $SeriesConfigs) {
        $pen = New-Object System.Drawing.Pen($series.Color, 3)
        $y = $legendY + ($idx * 18)
        $Graphics.DrawLine($pen, $legendX, $y + 6, $legendX + 18, $y + 6)
        $Graphics.DrawString($series.Name, $legendFont, [System.Drawing.Brushes]::Black, $legendX + 24, $y - 2)
        $idx++
    }
}

function Draw-EventPanel {
    param(
        $Graphics,
        [System.Drawing.Rectangle]$Rect,
        [string]$Title,
        [array]$SeriesConfigs,
        [int[]]$Years,
        [double]$YMin,
        [double]$YMax
    )

    $titleFont = New-Font -Size 14 -Style Bold
    $labelFont = New-Font -Size 10
    $axisPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(60,60,60), 1.5)
    $gridPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(225,225,225), 1)
    $gridPen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash
    $zeroPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(140,140,140), 1.5)
    $zeroPen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash

    $plot = [System.Drawing.Rectangle]::new($Rect.X + 70, $Rect.Y + 45, $Rect.Width - 110, $Rect.Height - 100)
    $Graphics.DrawString($Title, $titleFont, [System.Drawing.Brushes]::Black, $Rect.X + 8, $Rect.Y + 8)
    $Graphics.DrawRectangle($axisPen, $plot)

    $tickCount = 6
    for ($i = 0; $i -le $tickCount; $i++) {
        $yValue = $YMin + ($YMax - $YMin) * $i / $tickCount
        $py = $plot.Bottom - ($plot.Height * $i / $tickCount)
        $Graphics.DrawLine($gridPen, $plot.X, $py, $plot.Right, $py)
        $Graphics.DrawString(([Math]::Round($yValue,1)).ToString(), $labelFont, [System.Drawing.Brushes]::DimGray, $Rect.X + 8, $py - 8)
    }

    $zeroY = $plot.Bottom - ((0 - $YMin) / ($YMax - $YMin) * $plot.Height)
    $Graphics.DrawLine($zeroPen, $plot.X, $zeroY, $plot.Right, $zeroY)

    $nYears = $Years.Count
    for ($i = 0; $i -lt $nYears; $i++) {
        $x = $plot.X + ($plot.Width * $i / ($nYears - 1))
        $Graphics.DrawLine($axisPen, $x, $plot.Bottom, $x, $plot.Bottom + 4)
        $Graphics.DrawString($Years[$i].ToString(), $labelFont, [System.Drawing.Brushes]::DimGray, $x - 14, $plot.Bottom + 8)
    }

    foreach ($series in $SeriesConfigs) {
        $pen = New-Object System.Drawing.Pen($series.Color, 3)
        $brush = New-Object System.Drawing.SolidBrush($series.Color)
        $prevPoint = $null
        foreach ($year in $Years) {
            $row = $series.Data | Where-Object { [int]$_.year -eq $year } | Select-Object -First 1
            $estimate = Convert-ToNumber $row.estimate
            $lower = Convert-ToNumber $row.ci95_lower
            $upper = Convert-ToNumber $row.ci95_upper
            $x = $plot.X + ($plot.Width * ([array]::IndexOf($Years, $year)) / ($nYears - 1))
            $y = $plot.Bottom - (($estimate - $YMin) / ($YMax - $YMin) * $plot.Height)
            $yl = $plot.Bottom - (($lower - $YMin) / ($YMax - $YMin) * $plot.Height)
            $yu = $plot.Bottom - (($upper - $YMin) / ($YMax - $YMin) * $plot.Height)
            $Graphics.DrawLine($pen, $x, $yl, $x, $yu)
            $Graphics.DrawLine($pen, $x - 4, $yl, $x + 4, $yl)
            $Graphics.DrawLine($pen, $x - 4, $yu, $x + 4, $yu)
            if ($prevPoint) {
                $Graphics.DrawLine($pen, $prevPoint.X, $prevPoint.Y, $x, $y)
            }
            $Graphics.FillEllipse($brush, $x - 4, $y - 4, 8, 8)
            $prevPoint = [PSCustomObject]@{ X = $x; Y = $y }
        }
    }

    $legendX = $plot.Right - 120
    $legendY = $Rect.Y + 20
    $legendFont = New-Font -Size 9
    $idx = 0
    foreach ($series in $SeriesConfigs) {
        $pen = New-Object System.Drawing.Pen($series.Color, 3)
        $y = $legendY + ($idx * 18)
        $Graphics.DrawLine($pen, $legendX, $y + 6, $legendX + 18, $y + 6)
        $Graphics.DrawString($series.Name, $legendFont, [System.Drawing.Brushes]::Black, $legendX + 24, $y - 2)
        $idx++
    }
}

function Draw-StackedBars {
    param(
        $Graphics,
        [System.Drawing.Rectangle]$Rect,
        [array]$Rows
    )

    $titleFont = New-Font -Size 16 -Style Bold
    $labelFont = New-Font -Size 10
    $axisPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(60,60,60), 1.5)
    $Graphics.DrawString('Figure 3. Recovery classification in 2024 under two definitions', $titleFont, [System.Drawing.Brushes]::Black, $Rect.X + 8, $Rect.Y + 8)

    $plot = [System.Drawing.Rectangle]::new($Rect.X + 100, $Rect.Y + 60, $Rect.Width - 180, $Rect.Height - 130)
    $Graphics.DrawRectangle($axisPen, $plot)

    $classes = @(
        'Recovered on both definitions',
        'Only below 2019',
        'Only below expected range',
        'Not recovered on both definitions'
    )
    $colors = @{
        'Recovered on both definitions' = [System.Drawing.Color]::FromArgb(53, 114, 165)
        'Only below 2019' = [System.Drawing.Color]::FromArgb(238, 168, 51)
        'Only below expected range' = [System.Drawing.Color]::FromArgb(109, 182, 133)
        'Not recovered on both definitions' = [System.Drawing.Color]::FromArgb(193, 74, 68)
    }

    for ($i = 0; $i -le 4; $i++) {
        $y = $plot.Bottom - ($plot.Height * $i / 4)
        $Graphics.DrawLine((New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(225,225,225),1)), $plot.X, $y, $plot.Right, $y)
        $Graphics.DrawString((25 * $i).ToString(), $labelFont, [System.Drawing.Brushes]::DimGray, $Rect.X + 50, $y - 8)
    }

    $outcomes = @('dtp3','mcv1')
    for ($i = 0; $i -lt $outcomes.Count; $i++) {
        $outcome = $outcomes[$i]
        $x = $plot.X + 120 + ($i * 260)
        $barWidth = 120
        $cum = 0.0
        foreach ($class in $classes) {
            $row = $Rows | Where-Object { $_.outcome -eq $outcome -and $_.recovery_class_4 -eq $class } | Select-Object -First 1
            $share = Convert-ToNumber $row.share
            $height = $plot.Height * $share
            $y = $plot.Bottom - ($plot.Height * $cum) - $height
            $brush = New-Object System.Drawing.SolidBrush($colors[$class])
            $Graphics.FillRectangle($brush, $x, $y, $barWidth, $height)
            $cum += $share
        }
        $label = if ($outcome -eq 'dtp3') { 'DTP3' } else { 'MCV1' }
        $Graphics.DrawString($label, (New-Font -Size 12 -Style Bold), [System.Drawing.Brushes]::Black, $x + 28, $plot.Bottom + 10)
    }

    $legendFont = New-Font -Size 10
    $legendX = $plot.Right - 260
    $legendY = $Rect.Y + 70
    $idx = 0
    foreach ($class in $classes) {
        $brush = New-Object System.Drawing.SolidBrush($colors[$class])
        $y = $legendY + ($idx * 22)
        $Graphics.FillRectangle($brush, $legendX, $y, 14, 14)
        $Graphics.DrawString($class, $legendFont, [System.Drawing.Brushes]::Black, $legendX + 22, $y - 2)
        $idx++
    }
}

$globalMeans = Import-Csv (Join-Path $analysisDir 'descriptive_yearly_global_means.csv')
$eventStudy = Import-Csv (Join-Path $analysisDir 'event_study_main_fe.csv')
$recoverySummary = Import-Csv (Join-Path $analysisDir 'recovery_definition_summary.csv')

foreach ($file in @(
    'descriptive_overall_summary.csv',
    'event_study_main_fe.csv',
    'recovery_definition_summary.csv',
    'modified_poisson_rr_results.csv'
)) {
    Copy-Item -LiteralPath (Join-Path $analysisDir $file) -Destination (Join-Path $tableDir $file) -Force
}

$years = 2016..2024

$bmp1 = New-Object System.Drawing.Bitmap 1800, 900
$g1 = [System.Drawing.Graphics]::FromImage($bmp1)
$g1.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g1.Clear([System.Drawing.Color]::White)
Draw-PanelLabel -Graphics $g1 -Text 'A' -X 20 -Y 20
Draw-LinePanel -Graphics $g1 -Rect ([System.Drawing.Rectangle]::new(20,20,860,820)) -Title 'Figure 1A. Global DTP3 coverage, 2016-2024' -SeriesConfigs @(
    @{ Name = 'Unweighted'; Color = [System.Drawing.Color]::FromArgb(53,114,165); Data = @($globalMeans | Where-Object { $_.variable -eq 'dtp3' -and $_.weighting -eq 'unweighted' }) },
    @{ Name = 'Weighted by live births'; Color = [System.Drawing.Color]::FromArgb(193,74,68); Data = @($globalMeans | Where-Object { $_.variable -eq 'dtp3' -and $_.weighting -eq 'weighted' }) }
) -Years $years -YMin 75 -YMax 92 -YLabel 'Coverage (%)'
Draw-PanelLabel -Graphics $g1 -Text 'B' -X 900 -Y 20
Draw-LinePanel -Graphics $g1 -Rect ([System.Drawing.Rectangle]::new(900,20,860,820)) -Title 'Figure 1B. Global MCV1 coverage, 2016-2024' -SeriesConfigs @(
    @{ Name = 'Unweighted'; Color = [System.Drawing.Color]::FromArgb(53,114,165); Data = @($globalMeans | Where-Object { $_.variable -eq 'mcv1' -and $_.weighting -eq 'unweighted' }) },
    @{ Name = 'Weighted by live births'; Color = [System.Drawing.Color]::FromArgb(193,74,68); Data = @($globalMeans | Where-Object { $_.variable -eq 'mcv1' -and $_.weighting -eq 'weighted' }) }
) -Years $years -YMin 75 -YMax 90 -YLabel 'Coverage (%)'
$bmp1.Save((Join-Path $figDir 'figure1_global_trends.png'), [System.Drawing.Imaging.ImageFormat]::Png)
$g1.Dispose()
$bmp1.Dispose()

$eventYears = @(2016,2017,2018,2020,2021,2022,2023,2024)
$bmp2 = New-Object System.Drawing.Bitmap 1800, 900
$g2 = [System.Drawing.Graphics]::FromImage($bmp2)
$g2.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g2.Clear([System.Drawing.Color]::White)
Draw-PanelLabel -Graphics $g2 -Text 'A' -X 20 -Y 20
Draw-EventPanel -Graphics $g2 -Rect ([System.Drawing.Rectangle]::new(20,20,860,820)) -Title 'Figure 2A. Unweighted event-study coefficients' -SeriesConfigs @(
    @{ Name = 'DTP3'; Color = [System.Drawing.Color]::FromArgb(53,114,165); Data = @($eventStudy | Where-Object { $_.outcome -eq 'dtp3' -and $_.weighting -eq 'unweighted' }) },
    @{ Name = 'MCV1'; Color = [System.Drawing.Color]::FromArgb(193,74,68); Data = @($eventStudy | Where-Object { $_.outcome -eq 'mcv1' -and $_.weighting -eq 'unweighted' }) }
) -Years $eventYears -YMin -6.5 -YMax 2
Draw-PanelLabel -Graphics $g2 -Text 'B' -X 900 -Y 20
Draw-EventPanel -Graphics $g2 -Rect ([System.Drawing.Rectangle]::new(900,20,860,820)) -Title 'Figure 2B. Weighted event-study coefficients' -SeriesConfigs @(
    @{ Name = 'DTP3'; Color = [System.Drawing.Color]::FromArgb(53,114,165); Data = @($eventStudy | Where-Object { $_.outcome -eq 'dtp3' -and $_.weighting -eq 'weighted' }) },
    @{ Name = 'MCV1'; Color = [System.Drawing.Color]::FromArgb(193,74,68); Data = @($eventStudy | Where-Object { $_.outcome -eq 'mcv1' -and $_.weighting -eq 'weighted' }) }
) -Years $eventYears -YMin -6.5 -YMax 2
$bmp2.Save((Join-Path $figDir 'figure2_event_study.png'), [System.Drawing.Imaging.ImageFormat]::Png)
$g2.Dispose()
$bmp2.Dispose()

$bmp3 = New-Object System.Drawing.Bitmap 1400, 900
$g3 = [System.Drawing.Graphics]::FromImage($bmp3)
$g3.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g3.Clear([System.Drawing.Color]::White)
Draw-StackedBars -Graphics $g3 -Rect ([System.Drawing.Rectangle]::new(20,20,1360,840)) -Rows @($recoverySummary)
$bmp3.Save((Join-Path $figDir 'figure3_recovery_classification.png'), [System.Drawing.Imaging.ImageFormat]::Png)
$g3.Dispose()
$bmp3.Dispose()

$readmePath = Join-Path $analysisDir 'png_and_tables_readme.md'
@(
    '# PNG Figures and Main Tables'
    ''
    "PNG figures folder: $figDir"
    "- figure1_global_trends.png"
    "- figure2_event_study.png"
    "- figure3_recovery_classification.png"
    ''
    "Main tables folder: $tableDir"
    "- descriptive_overall_summary.csv"
    "- event_study_main_fe.csv"
    "- recovery_definition_summary.csv"
    "- modified_poisson_rr_results.csv"
) -join "`r`n" | Set-Content -LiteralPath $readmePath -Encoding UTF8

@(
    "PNG figures: $figDir"
    "Main tables: $tableDir"
    "Readme: $readmePath"
) -join "`r`n"
