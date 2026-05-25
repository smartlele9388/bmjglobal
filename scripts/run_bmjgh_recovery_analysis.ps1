$ErrorActionPreference = 'Stop'

$root = 'C:\Users\adamk\Documents\New project\downloads\bmjgh_sources'
$panelPath = Join-Path $root 'analysis\bmjgh_main_panel_2016_2024.csv'
$outputDir = Join-Path $root 'analysis\recovery_analysis'

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Add-Type -TypeDefinition @"
using System;

public static class BmMatrix {
    public static double[,] Invert(double[,] matrix) {
        int n = matrix.GetLength(0);
        if (n != matrix.GetLength(1)) throw new ArgumentException("Matrix must be square.");
        double[,] aug = new double[n, n * 2];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) aug[i, j] = matrix[i, j];
            aug[i, n + i] = 1.0;
        }
        for (int col = 0; col < n; col++) {
            int pivotRow = col;
            double pivotAbs = Math.Abs(aug[pivotRow, col]);
            for (int r = col + 1; r < n; r++) {
                double candidate = Math.Abs(aug[r, col]);
                if (candidate > pivotAbs) { pivotAbs = candidate; pivotRow = r; }
            }
            if (pivotAbs < 1e-12) throw new ArgumentException("Matrix is singular or near-singular.");
            if (pivotRow != col) {
                for (int c = 0; c < n * 2; c++) {
                    double tmp = aug[col, c];
                    aug[col, c] = aug[pivotRow, c];
                    aug[pivotRow, c] = tmp;
                }
            }
            double pivot = aug[col, col];
            for (int c = 0; c < n * 2; c++) aug[col, c] /= pivot;
            for (int r = 0; r < n; r++) {
                if (r == col) continue;
                double factor = aug[r, col];
                if (Math.Abs(factor) < 1e-16) continue;
                for (int c = 0; c < n * 2; c++) aug[r, c] -= factor * aug[col, c];
            }
        }
        double[,] inverse = new double[n, n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) inverse[i, j] = aug[i, n + j];
        }
        return inverse;
    }

    public static double[] MultiplyMV(double[,] matrix, double[] vector) {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        if (cols != vector.Length) throw new ArgumentException("Matrix/vector dimensions do not align.");
        double[] result = new double[rows];
        for (int i = 0; i < rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < cols; j++) sum += matrix[i, j] * vector[j];
            result[i] = sum;
        }
        return result;
    }

    public static double[,] MultiplyMM(double[,] a, double[,] b) {
        int aRows = a.GetLength(0);
        int aCols = a.GetLength(1);
        int bRows = b.GetLength(0);
        int bCols = b.GetLength(1);
        if (aCols != bRows) throw new ArgumentException("Matrix dimensions do not align.");
        double[,] result = new double[aRows, bCols];
        for (int i = 0; i < aRows; i++) {
            for (int k = 0; k < aCols; k++) {
                double aik = a[i, k];
                if (Math.Abs(aik) < 1e-16) continue;
                for (int j = 0; j < bCols; j++) result[i, j] += aik * b[k, j];
            }
        }
        return result;
    }
}
"@

function Convert-ToNumber {
    param($Value)

    if ($null -eq $Value) { return $null }
    $text = $Value.ToString().Trim()
    if ([string]::IsNullOrWhiteSpace($text)) { return $null }
    if ($text -in @('NA', 'N/A', '..', '...')) { return $null }
    $parsed = 0.0
    if ([double]::TryParse($text, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$parsed)) {
        return $parsed
    }

    return $null
}

function Get-WeightedMean {
    param(
        [double[]]$Values,
        [double[]]$Weights
    )

    if ($Values.Count -eq 0) { return $null }

    $sumW = 0.0
    $sumWX = 0.0
    for ($i = 0; $i -lt $Values.Count; $i++) {
        $w = $Weights[$i]
        $x = $Values[$i]
        if ($w -le 0) { continue }
        $sumW += $w
        $sumWX += $w * $x
    }

    if ($sumW -eq 0) { return $null }
    return $sumWX / $sumW
}

function Get-UnweightedMean {
    param([double[]]$Values)

    if ($Values.Count -eq 0) { return $null }
    return ($Values | Measure-Object -Average).Average
}

function Get-UnweightedSd {
    param([double[]]$Values)

    $n = $Values.Count
    if ($n -lt 2) { return $null }
    $mean = Get-UnweightedMean -Values $Values
    $sumSq = 0.0
    foreach ($x in $Values) {
        $sumSq += [Math]::Pow($x - $mean, 2)
    }
    return [Math]::Sqrt($sumSq / ($n - 1))
}

function Get-WeightedSd {
    param(
        [double[]]$Values,
        [double[]]$Weights
    )

    if ($Values.Count -eq 0) { return $null }
    $mean = Get-WeightedMean -Values $Values -Weights $Weights
    if ($null -eq $mean) { return $null }

    $sumW = 0.0
    $sumVar = 0.0
    for ($i = 0; $i -lt $Values.Count; $i++) {
        $w = $Weights[$i]
        $x = $Values[$i]
        if ($w -le 0) { continue }
        $sumW += $w
        $sumVar += $w * [Math]::Pow($x - $mean, 2)
    }

    if ($sumW -eq 0) { return $null }
    return [Math]::Sqrt($sumVar / $sumW)
}

function Get-WeightedCorrelation {
    param(
        [double[]]$X,
        [double[]]$Y,
        [double[]]$Weights
    )

    if (($X.Count -eq 0) -or ($Y.Count -eq 0)) { return $null }
    $meanX = Get-WeightedMean -Values $X -Weights $Weights
    $meanY = Get-WeightedMean -Values $Y -Weights $Weights
    if (($null -eq $meanX) -or ($null -eq $meanY)) { return $null }

    $sumW = 0.0
    $cov = 0.0
    $varX = 0.0
    $varY = 0.0

    for ($i = 0; $i -lt $X.Count; $i++) {
        $w = $Weights[$i]
        if ($w -le 0) { continue }
        $dx = $X[$i] - $meanX
        $dy = $Y[$i] - $meanY
        $sumW += $w
        $cov += $w * $dx * $dy
        $varX += $w * $dx * $dx
        $varY += $w * $dy * $dy
    }

    if (($sumW -eq 0) -or ($varX -le 0) -or ($varY -le 0)) { return $null }
    return $cov / [Math]::Sqrt($varX * $varY)
}

function Get-NormalCdf {
    param([double]$X)

    $sign = if ($X -lt 0) { -1.0 } else { 1.0 }
    $z = [Math]::Abs($X) / [Math]::Sqrt(2.0)
    $t = 1.0 / (1.0 + 0.3275911 * $z)
    $a1 = 0.254829592
    $a2 = -0.284496736
    $a3 = 1.421413741
    $a4 = -1.453152027
    $a5 = 1.061405429
    $erf = 1.0 - (((((($a5 * $t) + $a4) * $t) + $a3) * $t + $a2) * $t + $a1) * $t * [Math]::Exp(-1.0 * $z * $z)
    return 0.5 * (1.0 + $sign * $erf)
}

function Get-NormalPValue {
    param([double]$Z)

    return 2.0 * (1.0 - (Get-NormalCdf -X ([Math]::Abs($Z))))
}

function New-ZeroMatrix {
    param(
        [int]$Rows,
        [int]$Cols
    )

    return ,(New-Object 'double[,]' -ArgumentList $Rows, $Cols)
}

function Invert-Matrix {
    param($Matrix)

    if (($Matrix -is [object[]]) -and ($Matrix.Count -eq 1)) {
        $Matrix = $Matrix[0]
    }
    return ,([BmMatrix]::Invert([double[,]]$Matrix))
}

function Multiply-MatrixVector {
    param(
        $Matrix,
        [double[]]$Vector
    )

    if (($Matrix -is [object[]]) -and ($Matrix.Count -eq 1)) {
        $Matrix = $Matrix[0]
    }

    return [BmMatrix]::MultiplyMV([double[,]]$Matrix, $Vector)
}

function Multiply-Matrices {
    param(
        $A,
        $B
    )

    if (($A -is [object[]]) -and ($A.Count -eq 1)) {
        $A = $A[0]
    }
    if (($B -is [object[]]) -and ($B.Count -eq 1)) {
        $B = $B[0]
    }

    return ,([BmMatrix]::MultiplyMM([double[,]]$A, [double[,]]$B))
}

function Get-EventStudyRows {
    param(
        [object[]]$Rows,
        [string]$Outcome,
        [string]$Weighting,
        [string]$WeightField = 'live_births_approx'
    )

    $eventYears = @(2016, 2017, 2018, 2020, 2021, 2022, 2023, 2024)
    $groups = $Rows | Group-Object code
    $k = $eventYears.Count
    $starRows = New-Object System.Collections.Generic.List[object]

    foreach ($group in $groups) {
        $countryRows = $group.Group | Sort-Object year
        $wList = New-Object System.Collections.Generic.List[double]
        $yList = New-Object System.Collections.Generic.List[double]
        foreach ($row in $countryRows) {
            $w = if ($Weighting -eq 'weighted') { Convert-ToNumber $row.$WeightField } else { 1.0 }
            $y = Convert-ToNumber $row.$Outcome
            if (($null -eq $w) -or ($w -le 0) -or ($null -eq $y)) { continue }
            $wList.Add([double]$w)
            $yList.Add([double]$y)
        }
        if ($yList.Count -ne $countryRows.Count) {
            throw "Missing values encountered in event-study setup for $Outcome in country $($group.Name)."
        }

        $weights = $wList.ToArray()
        $yValues = $yList.ToArray()
        $sumW = ($weights | Measure-Object -Sum).Sum
        $yBar = 0.0
        for ($i = 0; $i -lt $weights.Count; $i++) {
            $yBar += $weights[$i] * $yValues[$i]
        }
        $yBar = $yBar / $sumW

        $xBar = @{}
        foreach ($eventYear in $eventYears) {
            $xBar[$eventYear] = 0.0
        }

        for ($i = 0; $i -lt $countryRows.Count; $i++) {
            $year = [int]$countryRows[$i].year
            if ($xBar.ContainsKey($year)) {
                $xBar[$year] += $weights[$i]
            }
        }
        foreach ($eventYear in $eventYears) {
            $xBar[$eventYear] = $xBar[$eventYear] / $sumW
        }

        for ($i = 0; $i -lt $countryRows.Count; $i++) {
            $row = $countryRows[$i]
            $w = $weights[$i]
            $scale = [Math]::Sqrt($w)
            $xStar = New-Object 'double[]' $k
            for ($j = 0; $j -lt $k; $j++) {
                $year = $eventYears[$j]
                $dummy = if ([int]$row.year -eq $year) { 1.0 } else { 0.0 }
                $xStar[$j] = ($dummy - $xBar[$year]) * $scale
            }

            $yStar = ((Convert-ToNumber $row.$Outcome) - $yBar) * $scale
            $starRows.Add([PSCustomObject]@{
                    code = $row.code
                    year = [int]$row.year
                    x = $xStar
                    y = $yStar
                })
        }
    }

    $n = $starRows.Count
    $xtx = New-ZeroMatrix -Rows $k -Cols $k
    $xty = New-Object 'double[]' $k
    foreach ($row in $starRows) {
        for ($a = 0; $a -lt $k; $a++) {
            $xty[$a] += $row.x[$a] * $row.y
            for ($b = 0; $b -lt $k; $b++) {
                $xtx.SetValue(($xtx.GetValue($a, $b) + $row.x[$a] * $row.x[$b]), $a, $b)
            }
        }
    }

    $bread = Invert-Matrix -Matrix $xtx
    $beta = Multiply-MatrixVector -Matrix $bread -Vector $xty

    $clusterScores = @{}
    foreach ($row in $starRows) {
        $fit = 0.0
        for ($j = 0; $j -lt $k; $j++) {
            $fit += $row.x[$j] * $beta[$j]
        }
        $resid = $row.y - $fit
        if (-not $clusterScores.ContainsKey($row.code)) {
            $clusterScores[$row.code] = New-Object 'double[]' $k
        }
        for ($j = 0; $j -lt $k; $j++) {
            $clusterScores[$row.code][$j] += $row.x[$j] * $resid
        }
    }

    $meat = New-ZeroMatrix -Rows $k -Cols $k
    foreach ($score in $clusterScores.Values) {
        for ($a = 0; $a -lt $k; $a++) {
            for ($b = 0; $b -lt $k; $b++) {
                $meat.SetValue(($meat.GetValue($a, $b) + $score[$a] * $score[$b]), $a, $b)
            }
        }
    }

    $breadMeat = Multiply-Matrices -A $bread -B $meat
    $vcov = Multiply-Matrices -A $breadMeat -B $bread
    $g = $clusterScores.Count
    $correction = ($g / ($g - 1.0)) * (($n - 1.0) / ($n - $k))
    for ($a = 0; $a -lt $k; $a++) {
        for ($b = 0; $b -lt $k; $b++) {
            $vcov.SetValue(($vcov.GetValue($a, $b) * $correction), $a, $b)
        }
    }

    $results = foreach ($j in 0..($k - 1)) {
        $estimate = $beta[$j]
        $se = [Math]::Sqrt([Math]::Max($vcov.GetValue($j, $j), 0.0))
        $z = if ($se -gt 0) { $estimate / $se } else { $null }
        $p = if ($null -ne $z) { Get-NormalPValue -Z $z } else { $null }
        [PSCustomObject][ordered]@{
            outcome = $Outcome
            weighting = $Weighting
            reference_year = 2019
            year = $eventYears[$j]
            estimate = [Math]::Round($estimate, 4)
            se_cluster_country = [Math]::Round($se, 4)
            ci95_lower = [Math]::Round($estimate - 1.96 * $se, 4)
            ci95_upper = [Math]::Round($estimate + 1.96 * $se, 4)
            z_stat = if ($null -ne $z) { [Math]::Round($z, 4) } else { $null }
            p_value_normal = if ($null -ne $p) { [Math]::Round($p, 6) } else { $null }
            n_obs = $n
            n_countries = $g
        }
    }

    return $results
}

function Get-TrendPredictionRow {
    param(
        [object[]]$CountryRows,
        [string]$Outcome
    )

    $trainYears = @(2016, 2017, 2018, 2019)
    $x = New-Object 'double[]' 4
    $y = New-Object 'double[]' 4
    for ($i = 0; $i -lt $trainYears.Count; $i++) {
        $year = $trainYears[$i]
        $row = $CountryRows | Where-Object { [int]$_.year -eq $year } | Select-Object -First 1
        $value = Convert-ToNumber $row.$Outcome
        if ($null -eq $value) {
            throw "Missing $Outcome for $($CountryRows[0].code) in year $year."
        }
        $x[$i] = [double]$year
        $y[$i] = [double]$value
    }

    $row2019 = $CountryRows | Where-Object { [int]$_.year -eq 2019 } | Select-Object -First 1
    $row2024 = $CountryRows | Where-Object { [int]$_.year -eq 2024 } | Select-Object -First 1
    $observed2019 = Convert-ToNumber $row2019.$Outcome
    $observed2024 = Convert-ToNumber $row2024.$Outcome

    $n = 4.0
    $xBar = ($x | Measure-Object -Average).Average
    $yBar = ($y | Measure-Object -Average).Average
    $sxx = 0.0
    $sxy = 0.0
    for ($i = 0; $i -lt 4; $i++) {
        $sxx += ($x[$i] - $xBar) * ($x[$i] - $xBar)
        $sxy += ($x[$i] - $xBar) * ($y[$i] - $yBar)
    }

    $slope = if ($sxx -ne 0) { $sxy / $sxx } else { 0.0 }
    $intercept = $yBar - $slope * $xBar
    $pred2024 = $intercept + $slope * 2024.0

    $sse = 0.0
    for ($i = 0; $i -lt 4; $i++) {
        $fit = $intercept + $slope * $x[$i]
        $sse += [Math]::Pow($y[$i] - $fit, 2)
    }

    $mse = $sse / 2.0
    $tCrit = 4.302652729911275
    $predSe = [Math]::Sqrt($mse * (1.0 + 1.0 / $n + [Math]::Pow(2024.0 - $xBar, 2) / $sxx))
    $piLower = $pred2024 - $tCrit * $predSe
    $piUpper = $pred2024 + $tCrit * $predSe

    $below2019 = if ($observed2024 -lt $observed2019) { 1 } else { 0 }
    $belowExpected = if ($observed2024 -lt $piLower) { 1 } else { 0 }
    $classification = switch ("$below2019|$belowExpected") {
        '0|0' { 'Recovered on both definitions' }
        '1|0' { 'Only below 2019' }
        '0|1' { 'Only below expected range' }
        '1|1' { 'Not recovered on both definitions' }
        default { 'Unknown' }
    }

    return [PSCustomObject][ordered]@{
        code = $CountryRows[0].code
        country_name = $CountryRows[0].country_name
        outcome = $Outcome
        observed_2019 = [Math]::Round($observed2019, 4)
        observed_2024 = [Math]::Round($observed2024, 4)
        change_2019_to_2024 = [Math]::Round($observed2024 - $observed2019, 4)
        trend_intercept = [Math]::Round($intercept, 6)
        trend_slope = [Math]::Round($slope, 6)
        predicted_2024 = [Math]::Round($pred2024, 4)
        predicted_lower_95pi_2024 = [Math]::Round($piLower, 4)
        predicted_upper_95pi_2024 = [Math]::Round($piUpper, 4)
        below2019 = $below2019
        below_expected_95pi = $belowExpected
        recovery_class_4 = $classification
    }
}

function Get-ModifiedPoissonResults {
    param(
        [object[]]$Rows,
        [string]$OutcomeField,
        [string]$OutcomeLabel
    )

    $analysisRows = $Rows | Where-Object {
        ($null -ne (Convert-ToNumber $_.$OutcomeField)) -and
        ($null -ne (Convert-ToNumber $_.prepandemic_mean_coverage)) -and
        ($null -ne (Convert-ToNumber $_.health_exp_gdp_pre)) -and
        ($null -ne (Convert-ToNumber $_.log_live_births_2019)) -and
        ($null -ne (Convert-ToNumber $_.climate_anomaly_mean_2020_2024_available))
    }

    $columnNames = @(
        'Intercept',
        'prepandemic_mean_coverage',
        'health_exp_gdp_pre',
        'log_live_births_2019',
        'climate_anomaly_mean_2020_2024_available',
        'fcs_any_2020_2024',
        'conflict_any_2020_2024',
        'income_low',
        'income_lower_middle',
        'income_upper_middle',
        'income_missing'
    )

    $n = $analysisRows.Count
    $k = $columnNames.Count
    if ($n -le $k) {
        throw "Not enough rows to fit modified Poisson for $OutcomeLabel."
    }

    $X = New-ZeroMatrix -Rows $n -Cols $k
    $Y = New-Object 'double[]' $n

    for ($i = 0; $i -lt $n; $i++) {
        $row = $analysisRows[$i]
        $income = $row.income_group_2019_2020
        $Y[$i] = [double](Convert-ToNumber $row.$OutcomeField)
        $X.SetValue(1.0, $i, 0)
        $X.SetValue([double](Convert-ToNumber $row.prepandemic_mean_coverage), $i, 1)
        $X.SetValue([double](Convert-ToNumber $row.health_exp_gdp_pre), $i, 2)
        $X.SetValue([double](Convert-ToNumber $row.log_live_births_2019), $i, 3)
        $X.SetValue([double](Convert-ToNumber $row.climate_anomaly_mean_2020_2024_available), $i, 4)
        $X.SetValue([double](Convert-ToNumber $row.fcs_any_2020_2024), $i, 5)
        $X.SetValue([double](Convert-ToNumber $row.conflict_any_2020_2024), $i, 6)
        $incomeLow = if ($income -eq 'Low income') { 1.0 } else { 0.0 }
        $incomeLowerMiddle = if ($income -eq 'Lower middle income') { 1.0 } else { 0.0 }
        $incomeUpperMiddle = if ($income -eq 'Upper middle income') { 1.0 } else { 0.0 }
        $incomeMissing = if ($income -eq 'Missing/Unknown') { 1.0 } else { 0.0 }
        $X.SetValue($incomeLow, $i, 7)
        $X.SetValue($incomeLowerMiddle, $i, 8)
        $X.SetValue($incomeUpperMiddle, $i, 9)
        $X.SetValue($incomeMissing, $i, 10)
    }

    $meanY = ($Y | Measure-Object -Average).Average
    if ($meanY -le 0) { $meanY = 0.01 }
    $beta = New-Object 'double[]' $k
    $beta[0] = [Math]::Log($meanY)

    $maxIter = 200
    for ($iter = 0; $iter -lt $maxIter; $iter++) {
        $xtwx = New-ZeroMatrix -Rows $k -Cols $k
        $xtwz = New-Object 'double[]' $k

        for ($i = 0; $i -lt $n; $i++) {
            $etaVal = 0.0
            for ($j = 0; $j -lt $k; $j++) {
                $etaVal += $X.GetValue($i, $j) * $beta[$j]
            }
            $etaVal = [Math]::Max([Math]::Min($etaVal, 20.0), -20.0)
            $muVal = [Math]::Exp($etaVal)
            if ($muVal -lt 1e-8) { $muVal = 1e-8 }
            $zVal = $etaVal + (($Y[$i] - $muVal) / $muVal)

            for ($a = 0; $a -lt $k; $a++) {
                $xtwz[$a] += $X.GetValue($i, $a) * $muVal * $zVal
                for ($b = 0; $b -lt $k; $b++) {
                    $xtwx.SetValue(($xtwx.GetValue($a, $b) + $X.GetValue($i, $a) * $muVal * $X.GetValue($i, $b)), $a, $b)
                }
            }
        }

        $xtwxInv = Invert-Matrix -Matrix $xtwx
        $betaNew = Multiply-MatrixVector -Matrix $xtwxInv -Vector $xtwz
        $maxDiff = 0.0
        for ($j = 0; $j -lt $k; $j++) {
            $diff = [Math]::Abs($betaNew[$j] - $beta[$j])
            if ($diff -gt $maxDiff) { $maxDiff = $diff }
        }
        $beta = $betaNew
        if ($maxDiff -lt 1e-8) { break }
    }

    $xtwxFinal = New-ZeroMatrix -Rows $k -Cols $k
    $muFinal = New-Object 'double[]' $n
    for ($i = 0; $i -lt $n; $i++) {
        $etaVal = 0.0
        for ($j = 0; $j -lt $k; $j++) {
            $etaVal += $X.GetValue($i, $j) * $beta[$j]
        }
        $etaVal = [Math]::Max([Math]::Min($etaVal, 20.0), -20.0)
        $muVal = [Math]::Exp($etaVal)
        if ($muVal -lt 1e-8) { $muVal = 1e-8 }
        $muFinal[$i] = $muVal
        for ($a = 0; $a -lt $k; $a++) {
            for ($b = 0; $b -lt $k; $b++) {
                $xtwxFinal.SetValue(($xtwxFinal.GetValue($a, $b) + $X.GetValue($i, $a) * $muVal * $X.GetValue($i, $b)), $a, $b)
            }
        }
    }

    $bread = Invert-Matrix -Matrix $xtwxFinal
    $meat = New-ZeroMatrix -Rows $k -Cols $k
    for ($i = 0; $i -lt $n; $i++) {
        $resid = $Y[$i] - $muFinal[$i]
        for ($a = 0; $a -lt $k; $a++) {
            for ($b = 0; $b -lt $k; $b++) {
                $meat.SetValue(($meat.GetValue($a, $b) + ($X.GetValue($i, $a) * $resid) * ($X.GetValue($i, $b) * $resid)), $a, $b)
            }
        }
    }

    $breadMeat = Multiply-Matrices -A $bread -B $meat
    $vcov = Multiply-Matrices -A $breadMeat -B $bread
    $correction = $n / ($n - $k)
    for ($a = 0; $a -lt $k; $a++) {
        for ($b = 0; $b -lt $k; $b++) {
            $vcov.SetValue(($vcov.GetValue($a, $b) * $correction), $a, $b)
        }
    }

    $results = foreach ($j in 0..($k - 1)) {
        $estimate = $beta[$j]
        $se = [Math]::Sqrt([Math]::Max($vcov.GetValue($j, $j), 0.0))
        $z = if ($se -gt 0) { $estimate / $se } else { $null }
        $p = if ($null -ne $z) { Get-NormalPValue -Z $z } else { $null }
        [PSCustomObject][ordered]@{
            model = $OutcomeLabel
            term = $columnNames[$j]
            estimate_log_rr = [Math]::Round($estimate, 6)
            se_robust = [Math]::Round($se, 6)
            rr = [Math]::Round([Math]::Exp($estimate), 4)
            ci95_lower_rr = [Math]::Round([Math]::Exp($estimate - 1.96 * $se), 4)
            ci95_upper_rr = [Math]::Round([Math]::Exp($estimate + 1.96 * $se), 4)
            z_stat = if ($null -ne $z) { [Math]::Round($z, 4) } else { $null }
            p_value_normal = if ($null -ne $p) { [Math]::Round($p, 6) } else { $null }
            n = $n
            events = ($Y | Measure-Object -Sum).Sum
        }
    }

    return $results
}

$panel = Import-Csv $panelPath | ForEach-Object {
    $dtp1 = Convert-ToNumber $_.dtp1
    $dtp3 = Convert-ToNumber $_.dtp3
    [PSCustomObject][ordered]@{
        code = $_.code
        country_name = $_.country_name
        year = [int]$_.year
        who_region = $_.who_region
        dtp1 = $dtp1
        dtp3 = $dtp3
        mcv1 = Convert-ToNumber $_.mcv1
        live_births_approx = Convert-ToNumber $_.live_births_approx
        health_exp_gdp = Convert-ToNumber $_.health_exp_gdp
        income_group = if ([string]::IsNullOrWhiteSpace($_.income_group)) { 'Missing/Unknown' } else { $_.income_group }
        fcs = Convert-ToNumber $_.fcs
        conflict = Convert-ToNumber $_.conflict
        conflict_deaths_best = Convert-ToNumber $_.conflict_deaths_best
        climate_anomaly = Convert-ToNumber $_.climate_anomaly
        dropout_abs = if (($null -ne $dtp1) -and ($null -ne $dtp3)) { $dtp1 - $dtp3 } else { $null }
        dropout_pct = if (($null -ne $dtp1) -and ($null -ne $dtp3) -and ($dtp1 -gt 0)) { (($dtp1 - $dtp3) / $dtp1) * 100.0 } else { $null }
    }
}

$analysisPanelPath = Join-Path $outputDir 'bmjgh_analysis_panel_with_dropout.csv'
$panel | Export-Csv -LiteralPath $analysisPanelPath -NoTypeInformation -Encoding UTF8

$summaryVariables = @('dtp1', 'dtp3', 'mcv1', 'dropout_abs', 'dropout_pct')
$overallSummary = foreach ($var in $summaryVariables) {
    $subset = $panel | Where-Object { $null -ne $_.$var }
    $values = [double[]]@($subset | ForEach-Object { [double]$_.$var })
    $weights = [double[]]@($subset | ForEach-Object { [double]$_.live_births_approx })
    [PSCustomObject][ordered]@{
        variable = $var
        n = $values.Count
        unweighted_mean = [Math]::Round((Get-UnweightedMean -Values $values), 4)
        unweighted_sd = [Math]::Round((Get-UnweightedSd -Values $values), 4)
        weighted_mean_live_births = [Math]::Round((Get-WeightedMean -Values $values -Weights $weights), 4)
        weighted_sd_live_births = [Math]::Round((Get-WeightedSd -Values $values -Weights $weights), 4)
    }
}
$overallSummaryPath = Join-Path $outputDir 'descriptive_overall_summary.csv'
$overallSummary | Export-Csv -LiteralPath $overallSummaryPath -NoTypeInformation -Encoding UTF8

$yearlyGlobalMeans = foreach ($year in 2016..2024) {
    $yearRows = $panel | Where-Object { $_.year -eq $year }
    foreach ($var in $summaryVariables) {
        $subset = $yearRows | Where-Object { $null -ne $_.$var }
        $values = [double[]]@($subset | ForEach-Object { [double]$_.$var })
        $weights = [double[]]@($subset | ForEach-Object { [double]$_.live_births_approx })
        foreach ($weighting in @('unweighted', 'weighted')) {
            [PSCustomObject][ordered]@{
                year = $year
                variable = $var
                weighting = $weighting
                mean = if ($weighting -eq 'weighted') {
                    [Math]::Round((Get-WeightedMean -Values $values -Weights $weights), 4)
                } else {
                    [Math]::Round((Get-UnweightedMean -Values $values), 4)
                }
                n = $values.Count
            }
        }
    }
}
$yearlyGlobalMeansPath = Join-Path $outputDir 'descriptive_yearly_global_means.csv'
$yearlyGlobalMeans | Export-Csv -LiteralPath $yearlyGlobalMeansPath -NoTypeInformation -Encoding UTF8

$regionYears = @(2019, 2021, 2024)
$regionVariables = @('dtp3', 'mcv1')
$regionMeans = foreach ($year in $regionYears) {
    foreach ($region in ($panel.who_region | Sort-Object -Unique)) {
        $regionRows = $panel | Where-Object { ($_.year -eq $year) -and ($_.who_region -eq $region) }
        foreach ($var in $regionVariables) {
            $subset = $regionRows | Where-Object { $null -ne $_.$var }
            $values = [double[]]@($subset | ForEach-Object { [double]$_.$var })
            $weights = [double[]]@($subset | ForEach-Object { [double]$_.live_births_approx })
            foreach ($weighting in @('unweighted', 'weighted')) {
                [PSCustomObject][ordered]@{
                    year = $year
                    who_region = $region
                    variable = $var
                    weighting = $weighting
                    mean = if ($weighting -eq 'weighted') {
                        [Math]::Round((Get-WeightedMean -Values $values -Weights $weights), 4)
                    } else {
                        [Math]::Round((Get-UnweightedMean -Values $values), 4)
                    }
                    n_countries = $subset.Count
                }
            }
        }
    }
}
$regionMeansPath = Join-Path $outputDir 'descriptive_region_year_means.csv'
$regionMeans | Export-Csv -LiteralPath $regionMeansPath -NoTypeInformation -Encoding UTF8

$corrValuesDtp3 = [double[]]@($panel | ForEach-Object { [double]$_.dtp3 })
$corrValuesMcv1 = [double[]]@($panel | ForEach-Object { [double]$_.mcv1 })
$corrWeights = [double[]]@($panel | ForEach-Object { [double]$_.live_births_approx })
$unitWeights = New-Object 'double[]' $corrValuesDtp3.Count
for ($i = 0; $i -lt $unitWeights.Count; $i++) { $unitWeights[$i] = 1.0 }
$corrUnweighted = [math]::Round((Get-WeightedCorrelation -X $corrValuesDtp3 -Y $corrValuesMcv1 -Weights $unitWeights), 4)
$corrWeighted = [math]::Round((Get-WeightedCorrelation -X $corrValuesDtp3 -Y $corrValuesMcv1 -Weights $corrWeights), 4)
$correlationTable = @(
    [PSCustomObject][ordered]@{ pair = 'dtp3_mcv1'; weighting = 'unweighted'; correlation = $corrUnweighted; n = $panel.Count }
    [PSCustomObject][ordered]@{ pair = 'dtp3_mcv1'; weighting = 'weighted'; correlation = $corrWeighted; n = $panel.Count }
)
$correlationPath = Join-Path $outputDir 'correlation_dtp3_mcv1.csv'
$correlationTable | Export-Csv -LiteralPath $correlationPath -NoTypeInformation -Encoding UTF8

$shockGapRows = foreach ($var in @('dtp3', 'mcv1')) {
    foreach ($weighting in @('unweighted', 'weighted')) {
        $year2019 = $panel | Where-Object { $_.year -eq 2019 }
        $year2021 = $panel | Where-Object { $_.year -eq 2021 }
        $year2024 = $panel | Where-Object { $_.year -eq 2024 }
        $v2019 = [double[]]@($year2019 | ForEach-Object { [double]$_.$var })
        $v2021 = [double[]]@($year2021 | ForEach-Object { [double]$_.$var })
        $v2024 = [double[]]@($year2024 | ForEach-Object { [double]$_.$var })
        $w2019 = [double[]]@($year2019 | ForEach-Object { [double]$_.live_births_approx })
        $w2021 = [double[]]@($year2021 | ForEach-Object { [double]$_.live_births_approx })
        $w2024 = [double[]]@($year2024 | ForEach-Object { [double]$_.live_births_approx })

        $m2019 = if ($weighting -eq 'weighted') { Get-WeightedMean -Values $v2019 -Weights $w2019 } else { Get-UnweightedMean -Values $v2019 }
        $m2021 = if ($weighting -eq 'weighted') { Get-WeightedMean -Values $v2021 -Weights $w2021 } else { Get-UnweightedMean -Values $v2021 }
        $m2024 = if ($weighting -eq 'weighted') { Get-WeightedMean -Values $v2024 -Weights $w2024 } else { Get-UnweightedMean -Values $v2024 }

        [PSCustomObject][ordered]@{
            variable = $var
            weighting = $weighting
            mean_2019 = [Math]::Round($m2019, 4)
            mean_2021 = [Math]::Round($m2021, 4)
            mean_2024 = [Math]::Round($m2024, 4)
            shock_2019_to_2021_pp = [Math]::Round($m2021 - $m2019, 4)
            change_2019_to_2024_pp = [Math]::Round($m2024 - $m2019, 4)
            recovery_gap_below_2019_pp = [Math]::Round($m2019 - $m2024, 4)
        }
    }
}
$shockGapPath = Join-Path $outputDir 'shock_and_recovery_gaps.csv'
$shockGapRows | Export-Csv -LiteralPath $shockGapPath -NoTypeInformation -Encoding UTF8

$eventStudyMain = @()
foreach ($outcome in @('dtp3', 'mcv1')) {
    $eventStudyMain += Get-EventStudyRows -Rows $panel -Outcome $outcome -Weighting 'unweighted'
    $eventStudyMain += Get-EventStudyRows -Rows $panel -Outcome $outcome -Weighting 'weighted'
}
$eventStudyMainPath = Join-Path $outputDir 'event_study_main_fe.csv'
$eventStudyMain | Export-Csv -LiteralPath $eventStudyMainPath -NoTypeInformation -Encoding UTF8

$eventStudyExtended = @()
foreach ($outcome in @('dtp1', 'dropout_abs', 'dropout_pct')) {
    $eventStudyExtended += Get-EventStudyRows -Rows $panel -Outcome $outcome -Weighting 'unweighted'
    $eventStudyExtended += Get-EventStudyRows -Rows $panel -Outcome $outcome -Weighting 'weighted'
}
$eventStudyExtendedPath = Join-Path $outputDir 'event_study_dtp1_dropout.csv'
$eventStudyExtended | Export-Csv -LiteralPath $eventStudyExtendedPath -NoTypeInformation -Encoding UTF8

$recoveryClassification = foreach ($group in ($panel | Group-Object code)) {
    $countryRows = $group.Group | Sort-Object year
    foreach ($outcome in @('dtp3', 'mcv1')) {
        Get-TrendPredictionRow -CountryRows $countryRows -Outcome $outcome
    }
}
$recoveryClassificationPath = Join-Path $outputDir 'country_recovery_classification.csv'
$recoveryClassification | Export-Csv -LiteralPath $recoveryClassificationPath -NoTypeInformation -Encoding UTF8

$recoverySummary = foreach ($outcome in @('dtp3', 'mcv1')) {
    $subset = $recoveryClassification | Where-Object { $_.outcome -eq $outcome }
    $total = $subset.Count
    $below2019Count = @($subset | Where-Object { [int]$_.below2019 -eq 1 }).Count
    $belowExpectedCount = @($subset | Where-Object { [int]$_.below_expected_95pi -eq 1 }).Count
    foreach ($class in @(
        'Recovered on both definitions',
        'Only below 2019',
        'Only below expected range',
        'Not recovered on both definitions'
    )) {
        $count = @($subset | Where-Object { $_.recovery_class_4 -eq $class }).Count
        [PSCustomObject][ordered]@{
            outcome = $outcome
            total_countries = $total
            below2019_count = $below2019Count
            below_expected_95pi_count = $belowExpectedCount
            recovery_class_4 = $class
            count = $count
            share = [Math]::Round($count / $total, 4)
        }
    }
}
$recoverySummaryPath = Join-Path $outputDir 'recovery_definition_summary.csv'
$recoverySummary | Export-Csv -LiteralPath $recoverySummaryPath -NoTypeInformation -Encoding UTF8

$driverRows = foreach ($group in ($panel | Group-Object code)) {
    $countryRows = $group.Group | Sort-Object year
    $row2019 = $countryRows | Where-Object { $_.year -eq 2019 } | Select-Object -First 1
    $row2020 = $countryRows | Where-Object { $_.year -eq 2020 } | Select-Object -First 1
    $births2019 = Convert-ToNumber $row2019.live_births_approx
    $healthPreValues = [double[]]@($countryRows | Where-Object { $_.year -in @(2017, 2018, 2019) -and $null -ne $_.health_exp_gdp } | ForEach-Object { [double]$_.health_exp_gdp })
    $climatePostValues = [double[]]@($countryRows | Where-Object { $_.year -in @(2020, 2021, 2022, 2023, 2024) -and $null -ne $_.climate_anomaly } | ForEach-Object { [double]$_.climate_anomaly })
    $incomeGroup = if (-not [string]::IsNullOrWhiteSpace($row2019.income_group)) { $row2019.income_group } elseif (-not [string]::IsNullOrWhiteSpace($row2020.income_group)) { $row2020.income_group } else { 'Missing/Unknown' }
    $fcsAny = @($countryRows | Where-Object { $_.year -in @(2020, 2021, 2022, 2023, 2024) } | ForEach-Object { [double]$_.fcs } | Measure-Object -Maximum).Maximum
    $conflictAny = @($countryRows | Where-Object { $_.year -in @(2020, 2021, 2022, 2023, 2024) } | ForEach-Object { [double]$_.conflict } | Measure-Object -Maximum).Maximum

    foreach ($outcome in @('dtp3', 'mcv1')) {
        $preMeanValues = [double[]]@($countryRows | Where-Object { $_.year -in @(2017, 2018, 2019) } | ForEach-Object { [double]$_.$outcome })
        $recoveryRow = $recoveryClassification | Where-Object { ($_.code -eq $group.Name) -and ($_.outcome -eq $outcome) } | Select-Object -First 1
        [PSCustomObject][ordered]@{
            code = $group.Name
            country_name = $countryRows[0].country_name
            outcome = $outcome
            prepandemic_mean_coverage = [Math]::Round((Get-UnweightedMean -Values $preMeanValues), 4)
            health_exp_gdp_pre = if ($healthPreValues.Count -gt 0) { [Math]::Round((Get-UnweightedMean -Values $healthPreValues), 4) } else { $null }
            income_group_2019_2020 = $incomeGroup
            fcs_any_2020_2024 = [int]$fcsAny
            conflict_any_2020_2024 = [int]$conflictAny
            climate_anomaly_mean_2020_2024_available = if ($climatePostValues.Count -gt 0) { [Math]::Round((Get-UnweightedMean -Values $climatePostValues), 4) } else { $null }
            live_births_2019 = if ($null -ne $births2019) { [Math]::Round($births2019, 4) } else { $null }
            log_live_births_2019 = if (($null -ne $births2019) -and ($births2019 -gt 0)) { [Math]::Round([Math]::Log($births2019), 6) } else { $null }
            observed_2019 = $recoveryRow.observed_2019
            observed_2024 = $recoveryRow.observed_2024
            not_recovered_2024 = [int]$recoveryRow.below2019
            below_expected_2024 = [int]$recoveryRow.below_expected_95pi
            predicted_2024 = $recoveryRow.predicted_2024
            predicted_lower_95pi_2024 = $recoveryRow.predicted_lower_95pi_2024
            recovery_class_4 = $recoveryRow.recovery_class_4
        }
    }
}
$driverDatasetPath = Join-Path $outputDir 'driver_dataset_long.csv'
$driverRows | Export-Csv -LiteralPath $driverDatasetPath -NoTypeInformation -Encoding UTF8

$poissonResults = @()
$poissonResults += Get-ModifiedPoissonResults -Rows ($driverRows | Where-Object { $_.outcome -eq 'dtp3' }) -OutcomeField 'not_recovered_2024' -OutcomeLabel 'dtp3_not_recovered_2024'
$poissonResults += Get-ModifiedPoissonResults -Rows ($driverRows | Where-Object { $_.outcome -eq 'mcv1' }) -OutcomeField 'not_recovered_2024' -OutcomeLabel 'mcv1_not_recovered_2024'
$poissonResults += Get-ModifiedPoissonResults -Rows ($driverRows | Where-Object { $_.outcome -eq 'dtp3' }) -OutcomeField 'below_expected_2024' -OutcomeLabel 'dtp3_below_expected_2024'
$poissonResults += Get-ModifiedPoissonResults -Rows ($driverRows | Where-Object { $_.outcome -eq 'mcv1' }) -OutcomeField 'below_expected_2024' -OutcomeLabel 'mcv1_below_expected_2024'
$poissonResultsPath = Join-Path $outputDir 'modified_poisson_rr_results.csv'
$poissonResults | Export-Csv -LiteralPath $poissonResultsPath -NoTypeInformation -Encoding UTF8

$dtp3Recovery = $recoveryClassification | Where-Object { $_.outcome -eq 'dtp3' }
$mcv1Recovery = $recoveryClassification | Where-Object { $_.outcome -eq 'mcv1' }
$dtp3Below2019 = @($dtp3Recovery | Where-Object { [int]$_.below2019 -eq 1 }).Count
$dtp3BelowExpected = @($dtp3Recovery | Where-Object { [int]$_.below_expected_95pi -eq 1 }).Count
$mcv1Below2019 = @($mcv1Recovery | Where-Object { [int]$_.below2019 -eq 1 }).Count
$mcv1BelowExpected = @($mcv1Recovery | Where-Object { [int]$_.below_expected_95pi -eq 1 }).Count

$summaryMdPath = Join-Path $outputDir 'analysis_summary.md'
$summaryLines = @(
    '# BMJGH Recovery Analysis'
    ''
    '## Outputs'
    "- Analysis panel with dropout: $analysisPanelPath"
    "- Overall descriptive summary: $overallSummaryPath"
    "- Yearly global means: $yearlyGlobalMeansPath"
    "- WHO region means (2019, 2021, 2024): $regionMeansPath"
    "- Correlation table: $correlationPath"
    "- Shock and recovery gap table: $shockGapPath"
    "- Main event-study FE results: $eventStudyMainPath"
    "- DTP1/dropout event-study results: $eventStudyExtendedPath"
    "- Country recovery classification: $recoveryClassificationPath"
    "- Recovery definition summary: $recoverySummaryPath"
    "- Driver dataset: $driverDatasetPath"
    "- Modified Poisson RR results: $poissonResultsPath"
    ''
    '## Key Recovery Counts'
    "- DTP3 below 2019 in 2024: $dtp3Below2019 countries"
    "- DTP3 below expected 95% PI in 2024: $dtp3BelowExpected countries"
    "- MCV1 below 2019 in 2024: $mcv1Below2019 countries"
    "- MCV1 below expected 95% PI in 2024: $mcv1BelowExpected countries"
    ''
    '## Notes'
    '- Weighted results use live_births_approx.'
    '- climate_anomaly mean for the driver model uses 2020-2024 available observed values; because the CCKP observed series currently ends in 2023, this is effectively the 2020-2023 mean.'
    '- Modified Poisson models use robust (sandwich) standard errors and report risk ratios.'
)
$summaryLines -join "`r`n" | Set-Content -LiteralPath $summaryMdPath -Encoding UTF8

@(
    "Analysis panel with dropout: $analysisPanelPath"
    "Main event-study FE: $eventStudyMainPath"
    "Recovery classification: $recoveryClassificationPath"
    "Modified Poisson RR: $poissonResultsPath"
    "Summary markdown: $summaryMdPath"
) -join "`r`n"
