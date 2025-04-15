/**
 * 伺服器液冷系統監控 - 前端邏輯
 * 處理數據獲取、表格更新、圖表渲染和用戶交互
 */

// 全局變量
let temperatureChart = null;
let controlChart = null;
let prevData = null; // 前一次的數據，用於計算趨勢

// 當頁面載入完成後初始化
document.addEventListener('DOMContentLoaded', () => {
    // 初始化圖表
    initCharts();
    
    // 設置目標溫度按鈕事件
    document.getElementById('set-gpu-target').addEventListener('click', updateGPUTarget);
    document.getElementById('set-cdu-target').addEventListener('click', updateCDUTarget);
    
    // 初始化表單
    fetch('/data')
        .then(response => response.json())
        .then(data => {
            document.getElementById('gpu-target').value = data.targets.GPU_target;
            document.getElementById('cdu-target').value = data.targets.target_temp;
            document.getElementById('cdu-target-display').textContent = data.targets.target_temp;
        });
    
    // 開始定期更新數據
    updateData();
    setInterval(updateData, 1000); // 每秒更新一次
    
    // 更新當前時間
    updateTime();
    setInterval(updateTime, 1000); // 每秒更新一次
    
    // 獲取歷史數據和更新圖表
    updateCharts();
    setInterval(updateCharts, 5000); // 每5秒更新一次圖表
    
    // 獲取 SA 優化日誌
    updateSALogs();
    setInterval(updateSALogs, 3000); // 每3秒更新一次日誌
});

/**
 * 初始化溫度和控制圖表
 */
function initCharts() {
    // 溫度圖表
    const temperatureCtx = document.getElementById('temperature-chart').getContext('2d');
    temperatureChart = new Chart(temperatureCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'GPU溫度 (°C)',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'CDU出口溫度 (°C)',
                    data: [],
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                title: {
                    display: true,
                    text: '溫度變化趨勢'
                },
                legend: {
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: '溫度 (°C)'
                    }
                }
            }
        }
    });
    
    // 控制圖表
    const controlCtx = document.getElementById('control-chart').getContext('2d');
    controlChart = new Chart(controlCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: '泵轉速 (%)',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    tension: 0.4,
                    fill: true,
                    yAxisID: 'y'
                },
                {
                    label: '風扇轉速 (%)',
                    data: [],
                    borderColor: 'rgb(153, 102, 255)',
                    backgroundColor: 'rgba(153, 102, 255, 0.1)',
                    tension: 0.4,
                    fill: true,
                    yAxisID: 'y'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                title: {
                    display: true,
                    text: '控制輸出變化'
                },
                legend: {
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: '轉速 (%)'
                    }
                }
            }
        }
    });
}

/**
 * 更新當前時間顯示
 */
function updateTime() {
    const now = new Date();
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');
    document.getElementById('current-time').textContent = `${hours}:${minutes}:${seconds}`;
}

/**
 * 獲取最新數據並更新界面
 */
function updateData() {
    fetch('/data')
        .then(response => response.json())
        .then(data => {
            updateTemperatureTable(data);
            updateControlTable(data);
            updateProgressBars(data);
            
            // 更新優化狀態
            const statusElement = document.getElementById('optimization-status');
            if (data.optimization_status) {
                statusElement.textContent = '優化中';
                statusElement.className = 'badge bg-warning float-end';
            } else {
                statusElement.textContent = '閒置';
                statusElement.className = 'badge bg-secondary float-end';
            }
            
            // 保存當前數據作為下一次的前一次數據
            prevData = data;
        })
        .catch(error => console.error('Error fetching data:', error));
}

/**
 * 更新溫度表格
 */
function updateTemperatureTable(data) {
    const temps = data.temperatures;
    const targets = data.targets;
    const table = document.getElementById('temperature-table');
    
    // 計算溫度差和狀態
    const gpuDiff = Math.abs(targets.GPU_target - temps.T_GPU).toFixed(1);
    const cduDiff = Math.abs(targets.target_temp - temps.T_CDU_out).toFixed(1);
    
    // 生成趨勢圖標
    const gpuTrend = getTrendIcon(temps.T_GPU, prevData ? prevData.temperatures.T_GPU : temps.T_GPU);
    const cduTrend = getTrendIcon(temps.T_CDU_out, prevData ? prevData.temperatures.T_CDU_out : temps.T_CDU_out);
    
    // 生成溫度狀態
    let gpuStatus = getTemperatureStatus(temps.T_GPU, targets.GPU_target);
    let cduStatus = getTemperatureStatus(temps.T_CDU_out, targets.target_temp);
    
    table.innerHTML = `
        <tr>
            <td><i class="fas fa-microchip"></i> GPU</td>
            <td>${temps.T_GPU.toFixed(1)}°C</td>
            <td>${gpuTrend}</td>
            <td>${gpuStatus} 目標: ${targets.GPU_target}°C (差: ${gpuDiff}°C)</td>
        </tr>
        <tr>
            <td><i class="fas fa-tint"></i> CDU出口</td>
            <td>${temps.T_CDU_out.toFixed(1)}°C</td>
            <td>${cduTrend}</td>
            <td>${cduStatus} 目標: ${targets.target_temp}°C (差: ${cduDiff}°C)</td>
        </tr>
        <tr>
            <td><i class="fas fa-cloud"></i> 環境</td>
            <td>${temps.T_env.toFixed(1)}°C</td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td><i class="fas fa-wind"></i> 空氣入/出</td>
            <td>${temps.T_air_in.toFixed(1)}°C / ${temps.T_air_out.toFixed(1)}°C</td>
            <td></td>
            <td>差: ${Math.abs(temps.T_air_out - temps.T_air_in).toFixed(1)}°C</td>
        </tr>
    `;
}

/**
 * 更新控制表格
 */
function updateControlTable(data) {
    const controls = data.controls;
    const table = document.getElementById('control-table');
    
    // 生成趨勢圖標
    const pumpTrend = getTrendIcon(controls.pump_duty, prevData ? prevData.controls.pump_duty : controls.pump_duty);
    const fanTrend = getTrendIcon(controls.fan_duty, prevData ? prevData.controls.fan_duty : controls.fan_duty);
    
    table.innerHTML = `
        <tr>
            <td><i class="fas fa-tint"></i> 泵轉速</td>
            <td>${controls.pump_duty}%</td>
            <td>${pumpTrend}</td>
            <td>調節中...</td>
        </tr>
        <tr>
            <td><i class="fas fa-fan"></i> 風扇轉速</td>
            <td>${controls.fan_duty}%</td>
            <td>${fanTrend}</td>
            <td>等待優化...</td>
        </tr>
    `;
}

/**
 * 更新進度條
 */
function updateProgressBars(data) {
    const controls = data.controls;
    
    // 泵進度條
    const pumpProgress = document.getElementById('pump-progress');
    pumpProgress.style.width = `${controls.pump_duty}%`;
    pumpProgress.textContent = `${controls.pump_duty}%`;
    pumpProgress.setAttribute('aria-valuenow', controls.pump_duty);
    
    // 風扇進度條
    const fanProgress = document.getElementById('fan-progress');
    fanProgress.style.width = `${controls.fan_duty}%`;
    fanProgress.textContent = `${controls.fan_duty}%`;
    fanProgress.setAttribute('aria-valuenow', controls.fan_duty);
}

/**
 * 獲取歷史數據並更新圖表
 */
function updateCharts() {
    fetch('/history')
        .then(response => response.json())
        .then(data => {
            // 更新溫度圖表
            temperatureChart.data.labels = data.timestamps;
            temperatureChart.data.datasets[0].data = data.T_GPU;
            temperatureChart.data.datasets[1].data = data.T_CDU_out;
            temperatureChart.update();
            
            // 更新控制圖表
            controlChart.data.labels = data.timestamps;
            controlChart.data.datasets[0].data = data.pump_duty;
            controlChart.data.datasets[1].data = data.fan_duty;
            controlChart.update();
        })
        .catch(error => console.error('Error fetching history data:', error));
}

/**
 * 獲取SA優化日誌
 */
function updateSALogs() {
    fetch('/logs')
        .then(response => response.json())
        .then(data => {
            const logsElement = document.getElementById('sa-logs');
            if (data.logs.length > 0) {
                let formattedLogs = '';
                
                data.logs.forEach(log => {
                    // 根據日誌內容添加不同的顏色類
                    if (log.includes('初始解')) {
                        formattedLogs += `<span class="log-initial">${log}</span>\n`;
                    } else if (log.includes('接受')) {
                        formattedLogs += `<span class="log-accept">${log}</span>\n`;
                    } else if (log.includes('拒絕')) {
                        formattedLogs += `<span class="log-reject">${log}</span>\n`;
                    } else if (log.includes('發現更好的解')) {
                        formattedLogs += `<span class="log-better">${log}</span>\n`;
                    } else if (log.includes('最終解') || log.includes('最佳化完成')) {
                        formattedLogs += `<span class="log-final">${log}</span>\n`;
                    } else {
                        formattedLogs += `${log}\n`;
                    }
                });
                
                logsElement.innerHTML = formattedLogs;
                
                // 自動滾動到底部
                logsElement.scrollTop = logsElement.scrollHeight;
            } else {
                logsElement.textContent = '等待優化...';
            }
        })
        .catch(error => console.error('Error fetching logs:', error));
}

/**
 * 更新GPU目標溫度
 */
function updateGPUTarget() {
    const gpuTarget = document.getElementById('gpu-target').value;
    if (gpuTarget >= 60 && gpuTarget <= 85) {
        fetch('/update_target', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ GPU_target: gpuTarget })
        })
        .then(response => response.json())
        .then(data => {
            console.log('GPU target updated:', data);
            // 可以添加成功提示
        })
        .catch(error => console.error('Error updating GPU target:', error));
    } else {
        alert('GPU目標溫度必須在60°C至85°C之間');
    }
}

/**
 * 更新冷卻水目標溫度
 */
function updateCDUTarget() {
    const cduTarget = document.getElementById('cdu-target').value;
    if (cduTarget >= 20 && cduTarget <= 40) {
        fetch('/update_target', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ target_temp: cduTarget })
        })
        .then(response => response.json())
        .then(data => {
            console.log('CDU target updated:', data);
            document.getElementById('cdu-target-display').textContent = data.target_temp;
            // 可以添加成功提示
        })
        .catch(error => console.error('Error updating CDU target:', error));
    } else {
        alert('冷卻水目標溫度必須在20°C至40°C之間');
    }
}

/**
 * 獲取趨勢圖標
 */
function getTrendIcon(current, previous) {
    if (current > previous) {
        return '<i class="fas fa-arrow-up trend-up"></i>';
    } else if (current < previous) {
        return '<i class="fas fa-arrow-down trend-down"></i>';
    } else {
        return '<i class="fas fa-equals trend-stable"></i>';
    }
}

/**
 * 獲取溫度狀態HTML
 */
function getTemperatureStatus(current, target) {
    const diff = current - target;
    
    if (diff > 2) {
        return '<span class="temp-high"><i class="fas fa-exclamation-triangle"></i> 過高</span>';
    } else if (diff < -2) {
        return '<span class="temp-low"><i class="fas fa-snowflake"></i> 偏低</span>';
    } else {
        return '<span class="temp-normal"><i class="fas fa-check-circle"></i> 正常</span>';
    }
} 