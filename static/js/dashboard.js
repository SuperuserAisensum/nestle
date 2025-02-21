// dashboard.js - Nestlé SKU Inventory Dashboard

// Initialize socket connection
const socket = io();

// Global state
let currentPage = 1;
const pageSize = 10;
let totalEvents = 0;
let events = [];
let skuData = {};

// Charts references
let mainChart = null;
let marketShareChart = null;
let dailyCountChart = null;

// DOM elements
const eventsTableBody = document.getElementById('eventsTableBody');
const paginationInfo = document.getElementById('paginationInfo');
const startCount = document.getElementById('startCount');
const endCount = document.getElementById('endCount');
const totalCount = document.getElementById('totalCount');
const prevPageBtn = document.getElementById('prevPage');
const nextPageBtn = document.getElementById('nextPage');
const eventFilter = document.getElementById('eventFilter');
const eventDetailModal = document.getElementById('eventDetailModal');
const closeModalBtn = document.getElementById('closeModal');
const dateRangeDisplay = document.getElementById('dateRangeDisplay');
const selectedDateDisplay = document.getElementById('selectedDate');

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    initializeDashboard();
    
    // Pagination controls
    prevPageBtn.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            fetchEvents();
        }
    });
    
    nextPageBtn.addEventListener('click', () => {
        if (currentPage * pageSize < totalEvents) {
            currentPage++;
            fetchEvents();
        }
    });
    
    // Event filter
    eventFilter.addEventListener('change', () => {
        currentPage = 1;
        fetchEvents();
    });
    
    // Modal close button
    closeModalBtn.addEventListener('click', () => {
        eventDetailModal.classList.add('hidden');
    });
    
    // Date selectors
    document.getElementById('dateSelector').addEventListener('click', () => {
        // For now, display today's date
        const today = new Date();
        selectedDateDisplay.textContent = `${today.toLocaleString('default', { month: 'short' })} ${today.getDate()}`;
    });
    
    // Navigation for daily count chart
    document.getElementById('prevDay').addEventListener('click', () => {
        // Would shift the date range back
        console.log("Previous day range");
    });
    
    document.getElementById('nextDay').addEventListener('click', () => {
        // Would shift the date range forward
        console.log("Next day range");
    });
});

// Socket event listeners
socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('new_detection', async (data) => {
    console.log('New detection received:', data);
    showToastNotification(`New detection from ${data.device_id}`);
    
    // Immediately update chart with new detection
    if (skuData.daily_data) {
        const dateIndex = skuData.daily_data.dates.indexOf(data.date);
        
        if (dateIndex !== -1) {
            // Update the values
            if (!skuData.daily_data.nestle_values[dateIndex]) {
                skuData.daily_data.nestle_values[dateIndex] = 0;
            }
            if (!skuData.daily_data.competitor_values[dateIndex]) {
                skuData.daily_data.competitor_values[dateIndex] = 0;
            }
            
            skuData.daily_data.nestle_values[dateIndex] += data.nestle_count;
            skuData.daily_data.competitor_values[dateIndex] += data.competitor_count;
            
            // Force chart update
            if (mainChart) {
                mainChart.data.datasets[0].data = skuData.daily_data.nestle_values;
                mainChart.data.datasets[1].data = skuData.daily_data.competitor_values;
                mainChart.update('active');
            }
        }
    }
    
    // Fetch complete fresh data
    await fetchDashboardData();
    
    // Update events list
    currentPage = 1;
    await fetchEvents();
});

// Initialize the dashboard
function initializeDashboard() {
    fetchDashboardData().then(() => {
        initializeChart();
        fetchEvents();
        populateDeviceFilter();
        startAutoRefresh();
    });
}

// Show toast notification
function showToastNotification(message) {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = 'toast-notification';
    toast.innerHTML = `
        <div class="flex items-center">
            <div class="flex-shrink-0 text-blue-500 mr-3">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd" />
                </svg>
            </div>
            <div>
                <p class="text-sm font-medium text-gray-900">${message}</p>
            </div>
        </div>
    `;
    
    // Add to body
    document.body.appendChild(toast);
    
    // Show the toast
    setTimeout(() => {
        toast.classList.add('show');
    }, 100);
    
    // Remove after 5 seconds
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            document.body.removeChild(toast);
        }, 300);
    }, 5000);
}

// Fetch dashboard data from server
async function fetchDashboardData() {
    try {
        const response = await fetch('/api/dashboard_data');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        skuData = data;
        
        updateDateRange();
        renderMainChart();
        renderMarketShareChart();
        renderDailyCountChart();
        updateStatisticsCards();
        
    } catch (error) {
        console.error("Error fetching dashboard data:", error);
    }
}

// Render events table
function renderEventsTable() {
    eventsTableBody.innerHTML = '';
    
    if (!events || !events.length) {
        eventsTableBody.innerHTML = `
            <tr>
                <td colspan="6" class="px-4 py-4 text-center text-gray-500">
                    No detection events found
                </td>
            </tr>
        `;
        return;
    }
    
    events.forEach(event => {
        const row = document.createElement('tr');
        row.className = 'hover:bg-gray-50';
        
        const total = event.nestle_detections + event.unclassified_detections;
        const nestlePercent = total > 0 ? Math.round((event.nestle_detections / total) * 100) : 0;
        
        row.innerHTML = `
            <td class="px-4 py-4 whitespace-nowrap text-sm text-gray-900">#${event.id}</td>
            <td class="px-4 py-4 whitespace-nowrap text-sm text-gray-500">${event.device_id}</td>
            <td class="px-4 py-4 whitespace-nowrap text-sm text-gray-500">${formatDate(event.timestamp)}</td>
            <td class="px-4 py-4 whitespace-nowrap">
                <span class="inline-flex items-center">
                    <span class="text-sm text-gray-900">${event.nestle_detections}</span>
                    <span class="ml-2 px-2 py-0.5 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">
                        ${nestlePercent}%
                    </span>
                </span>
            </td>
            <td class="px-4 py-4 whitespace-nowrap">
                <span class="inline-flex items-center">
                    <span class="text-sm text-gray-900">${event.unclassified_detections}</span>
                    <span class="ml-2 px-2 py-0.5 text-xs font-medium bg-red-100 text-red-800 rounded-full">
                        ${100 - nestlePercent}%
                    </span>
                </span>
            </td>
            <td class="px-4 py-4 whitespace-nowrap text-sm text-blue-600 hover:text-blue-900">
                <button onclick="viewEventDetails(${event.id})">
                    View
                </button>
            </td>
        `;
        
        eventsTableBody.appendChild(row);
    });
}

// Format date for display
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString();
}

// Update pagination information
function updatePagination() {
    const start = ((currentPage - 1) * pageSize) + 1;
    const end = Math.min(currentPage * pageSize, totalEvents);
    
    startCount.textContent = totalEvents > 0 ? start : 0;
    endCount.textContent = end;
    totalCount.textContent = totalEvents;
    
    // Enable/disable pagination buttons
    prevPageBtn.disabled = currentPage <= 1;
    prevPageBtn.classList.toggle('opacity-50', currentPage <= 1);
    
    nextPageBtn.disabled = currentPage * pageSize >= totalEvents;
    nextPageBtn.classList.toggle('opacity-50', currentPage * pageSize >= totalEvents);
}

// View event details - expose to global scope for onclick handler
window.viewEventDetails = async function(eventId) {
    try {
        const response = await fetch(`/api/events/${eventId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const eventData = await response.json();
        displayEventDetails(eventData);
        
    } catch (error) {
        console.error(`Error fetching event details for ID ${eventId}:`, error);
    }
};

// Display event details in modal
function displayEventDetails(eventData) {
    const modalEventInfo = document.getElementById('modalEventInfo');
    const modalNestleCount = document.getElementById('modalNestleCount');
    const modalCompCount = document.getElementById('modalCompCount');
    const modalTimestamp = document.getElementById('modalTimestamp');
    const detectedProducts = document.getElementById('detectedProducts');
    const eventImage = document.getElementById('eventImage').querySelector('img');
    
    // Format data for display
    const nestleCount = eventData.nestleCount || 0;
    const compCount = eventData.compCount || 0;
    const total = nestleCount + compCount;
    const nestlePercent = total > 0 ? Math.round((nestleCount / total) * 100) : 0;
    const compPercent = 100 - nestlePercent;
    
    // Update modal content
    modalEventInfo.textContent = `Event #${eventData.id} | Device: ${eventData.device_id}`;
    modalNestleCount.innerHTML = `
        ${nestleCount}
        <span class="ml-2 px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">${nestlePercent}%</span>
    `;
    modalCompCount.innerHTML = `
        ${compCount}
        <span class="ml-2 px-2 py-1 text-xs font-medium bg-red-100 text-red-800 rounded-full">${compPercent}%</span>
    `;
    modalTimestamp.textContent = formatDate(eventData.timestamp);
    
    // Set image source - prioritize labeled images
    if (eventData.labeled_images && eventData.labeled_images.length > 0) {
        // Use the first labeled image
        const labeledPath = eventData.labeled_images[0];
        eventImage.src = `/${labeledPath}`;
        eventImage.alt = `Labeled Detection Event #${eventData.id}`;
        console.log(`Using labeled image: ${labeledPath}`);
    } else if (eventData.image_path) {
        // Extract just the filename from the full path
        const pathParts = eventData.image_path.split('/');
        const deviceFolder = pathParts.length > 1 ? pathParts[pathParts.length - 2] + '/' : '';
        const filename = pathParts[pathParts.length - 1];
        
        // For labeled images, try to use the labeled version if available
        let labeledPath = eventData.image_path;
        if (eventData.image_path.includes('frame_')) {
            labeledPath = eventData.image_path.replace('frame_', 'labeled_');
        }
        
        // Try labeled image first
        if (labeledPath !== eventData.image_path) {
            const labeledParts = labeledPath.split('/');
            const labeledFilename = labeledParts[labeledParts.length - 1];
            eventImage.src = `/images/${deviceFolder}${labeledFilename}`;
            eventImage.alt = `Labeled Detection Event #${eventData.id}`;
            console.log(`Using derived labeled image: ${labeledFilename}`);
        } else {
            // Fall back to original image
            eventImage.src = `/images/${deviceFolder}${filename}`;
            eventImage.alt = `Detection Event #${eventData.id}`;
            console.log(`Using original image: ${filename}`);
        }
    } else {
        eventImage.src = '';
        eventImage.alt = 'No image available';
        console.log('No image available for this event');
    }
    
    // Populate product breakdown
    detectedProducts.innerHTML = '';
    if (eventData.products && eventData.products.length > 0) {
        eventData.products.forEach(product => {
            const productElement = document.createElement('div');
            productElement.className = 'flex items-center justify-between';
            productElement.innerHTML = `
                <span class="text-sm">${product.class || 'Unknown'}</span>
                <span class="text-sm font-medium">${(product.confidence * 100).toFixed(1)}%</span>
            `;
            detectedProducts.appendChild(productElement);
        });
    } else {
        detectedProducts.innerHTML = '<div class="text-sm text-gray-500">No detailed product data available</div>';
    }
    
    // Show modal
    eventDetailModal.classList.remove('hidden');
}

// Populate device filter
async function populateDeviceFilter() {
    try {
        const response = await fetch('/api/devices');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const devices = await response.json();
        
        // Add "All Devices" option
        if (!eventFilter.querySelector('option[value="all"]')) {
            const allOption = document.createElement('option');
            allOption.value = 'all';
            allOption.textContent = 'All Devices';
            eventFilter.appendChild(allOption);
        }
        
        // Add device options
        devices.forEach(device => {
            if (!eventFilter.querySelector(`option[value="${device.id}"]`)) {
                const option = document.createElement('option');
                option.value = device.id;
                option.textContent = device.name;
                eventFilter.appendChild(option);
            }
        });
    } catch (error) {
        console.error("Error fetching devices:", error);
    }
}

// Add auto-refresh functionality
function startAutoRefresh() {
    // Refresh every 30 seconds
    setInterval(async () => {
        await fetchDashboardData();
        if (currentPage === 1) {
            await fetchEvents();
        }
    }, 30000); // 30 seconds
}

// Fetch event data from server
async function fetchEvents() {
    try {
        let url = `/api/events?page=${currentPage}&limit=${pageSize}`;
        
        // Add filter if specified
        if (eventFilter.value !== 'all') {
            url += `&device_id=${eventFilter.value}`;
        }
        
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        events = data.data;
        totalEvents = data.pagination.total;
        
        renderEventsTable();
        updatePagination();
        
        // Update the showing count text
        const showingText = document.querySelector('.text-gray-500');
        if (showingText) {
            if (totalEvents > 0) {
                const start = ((currentPage - 1) * pageSize) + 1;
                const end = Math.min(currentPage * pageSize, totalEvents);
                showingText.textContent = `Showing ${start} to ${end} of ${totalEvents} events`;
            } else {
                showingText.textContent = 'No detection events found';
            }
        }
        
    } catch (error) {
        console.error("Error fetching events:", error);
        eventsTableBody.innerHTML = `
            <tr>
                <td colspan="6" class="px-4 py-4 text-center text-red-500">
                    Error loading detection events
                </td>
            </tr>
        `;
    }
}

// Update date range displays
function updateDateRange() {
    if (skuData.daily_data && skuData.daily_data.dates) {
        const firstDate = skuData.daily_data.dates[0];
        const lastDate = skuData.daily_data.dates[skuData.daily_data.dates.length - 1];
        
        dateRangeDisplay.textContent = `${firstDate} - ${lastDate}`;
        document.getElementById('dailyDateRange').textContent = `${firstDate} - ${lastDate}`;
        
        // Default selected date to latest date
        selectedDateDisplay.textContent = formatShortDate(lastDate);
    }
}

// Helper to format date as "Feb 20"
function formatShortDate(dateStr) {
    const date = new Date(dateStr);
    return `${date.toLocaleString('default', { month: 'short' })} ${date.getDate()}`;
}

// Render main chart comparing Nestlé vs Competitor products over time
function renderMainChart() {
    const ctx = document.getElementById('mainChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (mainChart) {
        mainChart.destroy();
    }
    
    // Prepare data
    let labels = [];
    let nestleData = [];
    let competitorData = [];
    
    if (skuData.daily_data) {
        labels = skuData.daily_data.dates || [];
        nestleData = skuData.daily_data.nestle_values || Array(labels.length).fill(0);
        competitorData = skuData.daily_data.competitor_values || Array(labels.length).fill(0);
    }
    
    mainChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Nestlé',
                    data: nestleData,
                    borderColor: '#3B82F6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Competitor',
                    data: competitorData,
                    borderColor: '#EF4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 750
            },
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    enabled: true,
                    mode: 'index',
                    intersect: false,
                    padding: 10,
                    backgroundColor: 'rgba(255, 255, 255, 0.9)',
                    titleColor: '#000',
                    titleFont: {
                        size: 14,
                        weight: 'bold'
                    },
                    bodyColor: '#666',
                    bodyFont: {
                        size: 13
                    },
                    borderColor: '#ddd',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 12
                        }
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        font: {
                            size: 12
                        },
                        stepSize: 1 // Force integer steps
                    }
                }
            }
        }
    });
}

// Render market share pie chart
function renderMarketShareChart() {
    const ctx = document.getElementById('marketShareChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (marketShareChart) {
        marketShareChart.destroy();
    }
    
    // Prepare data
    let nestleShare = 50;
    let competitorShare = 50;
    
    if (skuData.market_share && skuData.market_share.values) {
        nestleShare = skuData.market_share.values[0] || 50;
        competitorShare = skuData.market_share.values[1] || 50;
    }
    
    marketShareChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Nestlé', 'Competitor'],
            datasets: [{
                data: [nestleShare, competitorShare],
                backgroundColor: [
                    '#3B82F6', // Blue for Nestlé
                    '#EF4444'  // Red for Competitor
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%',
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.raw}%`;
                        }
                    }
                }
            }
        }
    });
}

// Render daily count chart
function renderDailyCountChart() {
    const ctx = document.getElementById('dailyCountChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (dailyCountChart) {
        dailyCountChart.destroy();
    }
    
    // Prepare data
    let labels = [];
    let countData = [];
    
    if (skuData.daily_count) {
        labels = skuData.daily_count.dates || [];
        countData = skuData.daily_count.counts || Array(labels.length).fill(0);
    }
    
    dailyCountChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Total Count',
                data: countData,
                backgroundColor: '#3B82F6',
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                }
            }
        }
    });
}

// Update statistics cards with real data
function updateStatisticsCards() {
    if (skuData.nestle) {
        // Update Nestlé statistics
        const nestleElements = document.querySelectorAll('.card:nth-child(1) .text-3xl');
        if (nestleElements.length >= 3) {
            nestleElements[0].textContent = skuData.nestle.max.count || 0;
            nestleElements[1].textContent = skuData.nestle.avg.count || 0;
            nestleElements[2].textContent = skuData.nestle.min.count || 0;
        }
        
        const nestleDates = document.querySelectorAll('.card:nth-child(1) .font-medium.text-gray-800');
        if (nestleDates.length >= 3) {
            nestleDates[0].textContent = skuData.nestle.max.date || '';
            nestleDates[1].textContent = skuData.nestle.avg.period || 'Last 7 days';
            nestleDates[2].textContent = skuData.nestle.min.date || '';
        }
    }
    
    if (skuData.competitor) {
        // Update Competitor statistics
        const compElements = document.querySelectorAll('.card:nth-child(2) .text-3xl');
        if (compElements.length >= 3) {
            compElements[0].textContent = skuData.competitor.max.count || 0;
            compElements[1].textContent = skuData.competitor.avg.count || 0;
            compElements[2].textContent = skuData.competitor.min.count || 0;
        }
        
        const compDates = document.querySelectorAll('.card:nth-child(2) .font-medium.text-gray-800');
        if (compDates.length >= 3) {
            compDates[0].textContent = skuData.competitor.max.date || '';
            compDates[1].textContent = skuData.competitor.avg.period || 'Last 7 days';
            compDates[2].textContent = skuData.competitor.min.date || '';
        }
    }
    
    // Update Top 3 Nestlé SKUs
    if (skuData.top_products && skuData.top_products.length > 0) {
        const topProductsCards = document.querySelector('.card .flex.justify-around');
        if (topProductsCards) {
            const productElements = topProductsCards.querySelectorAll('.flex.flex-col.items-center');
            
            // Get max count for scaling
            const maxCount = Math.max(...skuData.top_products.map(p => p.count || 0));
            
            skuData.top_products.forEach((product, index) => {
                if (index < productElements.length) {
                    const countElement = productElements[index].querySelector('.text-sm.font-medium');
                    const barElement = productElements[index].querySelector('.w-12');
                    const nameElement = productElements[index].querySelector('.mt-2');
                    
                    if (countElement) countElement.textContent = product.count || 0;
                    if (nameElement) nameElement.textContent = product.name || `Product ${index + 1}`;
                    
                    // Scale the bar height (max height = 168px)
                    if (barElement) {
                        const height = maxCount > 0 ? Math.round((product.count / maxCount) * 168) : 0;
                        barElement.style.height = `${Math.max(height, 10)}px`;
                    }
                }
            });
        }
    }
    
    // Update market share percentages in the legend
    if (skuData.market_share && skuData.market_share.values) {
        const nestleShare = skuData.market_share.values[0] || 50;
        const competitorShare = skuData.market_share.values[1] || 50;
        
        const marketShareLegend = document.querySelector('.flex.justify-center.space-x-10');
        if (marketShareLegend) {
            const percents = marketShareLegend.querySelectorAll('.block.text-sm.text-gray-500');
            if (percents.length >= 2) {
                percents[0].textContent = `${nestleShare}%`;
                percents[1].textContent = `${competitorShare}%`;
            }
        }
    }
}

// Update the image upload handler section
document.getElementById('imageInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const uploadStatus = document.getElementById('uploadStatus');
    const detectionResults = document.getElementById('detectionResults');
    
    uploadStatus.textContent = 'Processing image...';
    uploadStatus.classList.remove('text-red-500', 'text-green-500');
    uploadStatus.classList.add('text-gray-500');
    
    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('/check_image', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Failed to process image');
        }

        const result = await response.json();
        
        // Update counts
        document.getElementById('nestleCount').textContent = result.total_nestle;
        document.getElementById('competitorCount').textContent = result.total_competitor;
        
        // Update product breakdown
        const productList = document.getElementById('productList');
        productList.innerHTML = '';

        // Show Nestlé products (from Roboflow)
        if (Object.keys(result.nestle_products).length > 0) {
            const nestleHeader = document.createElement('div');
            nestleHeader.className = 'font-medium text-blue-700 mb-2';
            nestleHeader.textContent = 'Nestlé Products (Roboflow):';
            productList.appendChild(nestleHeader);
            
            Object.entries(result.nestle_products).forEach(([product, count]) => {
                const item = document.createElement('div');
                item.className = 'flex justify-between items-center text-sm pl-2 mb-1';
                item.innerHTML = `
                    <span class="text-gray-700">${product}</span>
                    <span class="font-medium bg-blue-50 px-2 py-1 rounded">${count}</span>
                `;
                productList.appendChild(item);
            });
        }

        // Show Competitor products (from DINO-X)
        if (Object.keys(result.competitor_products).length > 0) {
            if (productList.children.length > 0) {
                productList.appendChild(document.createElement('hr'));
            }
            
            const compHeader = document.createElement('div');
            compHeader.className = 'font-medium text-red-700 mt-4 mb-2';
            compHeader.textContent = 'Competitor Products (DINO-X):';
            productList.appendChild(compHeader);
            
            Object.entries(result.competitor_products).forEach(([product, count]) => {
                const item = document.createElement('div');
                item.className = 'flex justify-between items-center text-sm pl-2 mb-1';
                item.innerHTML = `
                    <span class="text-gray-700">${product}</span>
                    <span class="font-medium bg-red-50 px-2 py-1 rounded">${count}</span>
                `;
                productList.appendChild(item);
            });
        }

        if (productList.children.length === 0) {
            productList.innerHTML = '<div class="text-gray-500 text-sm">No products detected</div>';
        }
        
        // Display labeled image
        const labeledImage = document.getElementById('labeledImage');
        labeledImage.src = '/' + result.labeled_image;
        
        // Show results
        detectionResults.classList.remove('hidden');
        uploadStatus.textContent = 'Processing complete!';
        uploadStatus.classList.remove('text-gray-500');
        uploadStatus.classList.add('text-green-500');

        // Update dashboard data and charts after detection
        await updateDashboardAfterDetection(result);

    } catch (error) {
        console.error('Error:', error);
        uploadStatus.textContent = 'Error processing image: ' + error.message;
        uploadStatus.classList.remove('text-gray-500', 'text-green-500');
        uploadStatus.classList.add('text-red-500');
    }
});

// Modify updateDashboardAfterDetection function
async function updateDashboardAfterDetection(detectionResult) {
    try {
        const currentDate = detectionResult.date;
        
        // Immediately update local chart data
        if (skuData.daily_data) {
            const dateIndex = skuData.daily_data.dates.indexOf(currentDate);
            
            if (dateIndex !== -1) {
                // Update the values for today
                if (!skuData.daily_data.nestle_values[dateIndex]) {
                    skuData.daily_data.nestle_values[dateIndex] = 0;
                }
                if (!skuData.daily_data.competitor_values[dateIndex]) {
                    skuData.daily_data.competitor_values[dateIndex] = 0;
                }
                
                // Add new detection counts to existing values
                skuData.daily_data.nestle_values[dateIndex] += detectionResult.total_nestle;
                skuData.daily_data.competitor_values[dateIndex] += detectionResult.total_competitor;
                
                // Force immediate chart update
                if (mainChart) {
                    mainChart.data.datasets[0].data = skuData.daily_data.nestle_values;
                    mainChart.data.datasets[1].data = skuData.daily_data.competitor_values;
                    mainChart.update('active');
                }
            }
        }

        // Fetch fresh dashboard data
        const response = await fetch('/api/dashboard_data');
        if (!response.ok) {
            throw new Error('Failed to fetch updated dashboard data');
        }
        const freshData = await response.json();
        
        // Update local data
        skuData = freshData;

        // Update all visualizations
        updateDateRange();
        renderMainChart();
        renderMarketShareChart();
        renderDailyCountChart();
        updateStatisticsCards();

        // Refresh events table
        currentPage = 1;
        await fetchEvents();

    } catch (error) {
        console.error('Error updating dashboard:', error);
    }
}

// Drag and drop handling
const dropZone = document.querySelector('label[for="imageInput"]');

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults (e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, unhighlight, false);
});

function highlight(e) {
    dropZone.classList.add('border-blue-500', 'bg-blue-50');
}

function unhighlight(e) {
    dropZone.classList.remove('border-blue-500', 'bg-blue-50');
}

dropZone.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const file = dt.files[0];
    
    const input = document.getElementById('imageInput');
    input.files = dt.files;
    input.dispatchEvent(new Event('change'));
}

// Add this function to ensure chart is properly initialized
function initializeChart() {
    if (mainChart) {
        mainChart.destroy();
    }
    
    const ctx = document.getElementById('mainChart').getContext('2d');
    mainChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: skuData.daily_data.dates,
            datasets: [
                {
                    label: 'Nestlé',
                    data: skuData.daily_data.nestle_values,
                    borderColor: '#3B82F6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Competitor',
                    data: skuData.daily_data.competitor_values,
                    borderColor: '#EF4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 750
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
}