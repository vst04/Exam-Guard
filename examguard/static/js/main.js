// Common JavaScript functionality

// Initialize ScrollReveal for animations
const sr = ScrollReveal({
    origin: 'top',
    distance: '60px',
    duration: 2500,
    delay: 400
});

// Apply reveal animations to common elements
sr.reveal('.nav__logo', { delay: 100 });
sr.reveal('.nav__links', { delay: 200 });
sr.reveal('h1, h2', { delay: 300 });
sr.reveal('p, form', { delay: 400 });
sr.reveal('button', { delay: 500 });

// Handle mobile menu
document.addEventListener('DOMContentLoaded', function() {
    const menuBtn = document.getElementById('menu-btn');
    const navLinks = document.getElementById('nav-links');
    
    if (menuBtn && navLinks) {
        menuBtn.addEventListener('click', function() {
            navLinks.classList.toggle('open');
        });
    }
    
    // Close menu when clicking outside
    document.addEventListener('click', function(e) {
        if (navLinks && navLinks.classList.contains('open') && 
            !e.target.closest('.nav__links') && 
            !e.target.closest('.nav__menu__btn')) {
            navLinks.classList.remove('open');
        }
    });
});

// ScrollReveal options
const scrollRevealOption = {
  distance: "50px",
  origin: "bottom",
  duration: 1000,
};

if (typeof ScrollReveal !== "undefined") {
  ScrollReveal().reveal(".header__image img", {
    ...scrollRevealOption,
    origin: "right",
  });
  ScrollReveal().reveal(".header__content h2", {
    ...scrollRevealOption,
    delay: 500,
  });
  ScrollReveal().reveal(".header__content h1", {
    ...scrollRevealOption,
    delay: 1000,
  });
  ScrollReveal().reveal(".header__content p", {
    ...scrollRevealOption,
    delay: 1500,
  });
  ScrollReveal().reveal(".header__content .header__btn", {
    ...scrollRevealOption,
    delay: 2000,
  });
  ScrollReveal().reveal(".header__content .socials", {
    ...scrollRevealOption,
    delay: 2500,
  });
  ScrollReveal().reveal(".header__bar", {
    ...scrollRevealOption,
    delay: 3000,
  });
}

// Password visibility toggle
const passwordToggle = (inputId, eyeId) => {
  const input = document.getElementById(inputId);
  const eyeIcon = document.getElementById(eyeId);

  if (!input || !eyeIcon) return;

  eyeIcon.addEventListener("click", () => {
    input.type = input.type === "password" ? "text" : "password";
    eyeIcon.classList.toggle("ri-eye-fill");
    eyeIcon.classList.toggle("ri-eye-off-fill");
  });
};

// Video feed error handling function
function handleVideoError(img) {
  console.error("Video feed error");
  // Retry connection after 5 seconds
  setTimeout(() => {
      img.src = "http://localhost:5000/video_feed?" + new Date().getTime();
  }, 5000);
}
// Add this to your existing script
document.getElementById('video-feed').onerror = function() {
  handleVideoError(this);
};

document.addEventListener("DOMContentLoaded", function () {
  const loginContainer = document.getElementById("loginAccessRegister");
  const registerBtn = document.getElementById("loginButtonRegister");
  const loginBtn = document.getElementById("loginButtonAccess");
  const getStartedBtn = document.getElementById("getStartedButton");
  
  // Initialize video feed
  const videoFeed = document.querySelector('.camera-feed img');
  if (videoFeed) {
      videoFeed.src = "http://localhost:5000/video_feed?" + new Date().getTime();
  }

  // Show signup form when "Create Account" is clicked
  if (registerBtn && loginContainer) {
    registerBtn.addEventListener("click", () => {
      loginContainer.classList.add("active");
    });
  }

  // Show login form when "Log In" is clicked
  if (loginBtn && loginContainer) {
    loginBtn.addEventListener("click", () => {
      loginContainer.classList.remove("active");
    });
  }

  // Show signup form when "Get Started" is clicked
  if (getStartedBtn && loginContainer) {
    getStartedBtn.addEventListener("click", () => {
      loginContainer.classList.add("active");
    });
  }
});

// Additional ScrollReveal Animations
if (typeof ScrollReveal !== "undefined") {
  ScrollReveal().reveal(".login__title", {
    ...scrollRevealOption,
    delay: 100,
  });

  ScrollReveal().reveal(".login__box", {
    ...scrollRevealOption,
    delay: 200,
    interval: 100,
  });

  ScrollReveal().reveal(".login__button", {
    ...scrollRevealOption,
    delay: 400,
  });

  ScrollReveal().reveal(".login__social", {
    ...scrollRevealOption,
    delay: 600,
  });

  ScrollReveal().reveal(".login__switch", {
    ...scrollRevealOption,
    delay: 800,
  });
}

// Dashboard functionality
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Socket.IO connection
    const socket = io();

    // DOM Elements
    const mainVideo = document.getElementById('main-video');
    const pauseOverlay = document.getElementById('pause-overlay');
    const pauseBtn = document.getElementById('pause-btn');
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    const startMonitoringBtn = document.getElementById('start-monitoring');
    const stopMonitoringBtn = document.getElementById('stop-monitoring');
    const gridViewBtn = document.getElementById('grid-view-btn');
    const analyticsViewBtn = document.getElementById('analytics-view-btn');
    const studentGrid = document.getElementById('student-grid');
    const analyticsView = document.getElementById('analytics-view');
    const analyticsBody = document.getElementById('analytics-body');
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');
    const alertCount = document.getElementById('alert-count');
    const headTurnCount = document.getElementById('head-turn-count');
    const handGestureCount = document.getElementById('hand-gesture-count');
    const phoneCount = document.getElementById('phone-count');
    const activeStudents = document.getElementById('active-students');
    const riskFilter = document.getElementById('risk-filter');
    const sortBy = document.getElementById('sort-by');
    const studentSearch = document.getElementById('student-search');
    const studentList = document.getElementById('student-list');
    const totalViolations = document.getElementById('total-violations');
    const totalStudents = document.getElementById('total-students');
    const monitoringTime = document.getElementById('monitoring-time');
    const studentDetails = document.getElementById('student-details');

    // State management
    let isPaused = false;
    let isMonitoring = false;
    let studentData = new Map(); // Store student data for analytics
    let totalHeadTurns = 0;
    let totalHandGestures = 0;
    let totalPhoneDetections = 0;
    let monitoringStartTime = null;
    let monitoringTimer = null;
    let selectedStudentId = null;

    // Initialize counters
    if (headTurnCount) headTurnCount.textContent = '0';
    if (handGestureCount) handGestureCount.textContent = '0';
    if (phoneCount) phoneCount.textContent = '0';
    if (alertCount) alertCount.textContent = '0';
    if (activeStudents) activeStudents.textContent = '0';
    if (totalViolations) totalViolations.textContent = '0';
    if (totalStudents) totalStudents.textContent = '0';
    if (monitoringTime) monitoringTime.textContent = '00:00:00';

    // Tab Navigation
    if (navItems) {
        navItems.forEach(item => {
            item.addEventListener('click', () => {
                const tabId = item.getAttribute('data-tab');
                showTab(tabId);
                
                // Update active state
                navItems.forEach(nav => nav.classList.remove('active'));
                item.classList.add('active');
                
                // Update analytics data if switching to analytics tab
                if (tabId === 'analytics') {
                    updateAnalyticsSummary();
                }
            });
        });
    }

    function showTab(tabId) {
        if (tabContents) {
            tabContents.forEach(content => {
                if (content.id === tabId + '-tab') {
                    content.classList.remove('hidden');
                } else {
                    content.classList.add('hidden');
                }
            });
        }
    }

    // Video Controls
    if (pauseBtn) {
        pauseBtn.addEventListener('click', () => {
            isPaused = !isPaused;
            if (isPaused) {
                if (mainVideo) mainVideo.classList.add('paused');
                if (pauseOverlay) pauseOverlay.classList.remove('hidden');
                pauseBtn.innerHTML = '<i class="fas fa-play"></i>';
            } else {
                if (mainVideo) mainVideo.classList.remove('paused');
                if (pauseOverlay) pauseOverlay.classList.add('hidden');
                pauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
            }
        });
    }

    if (fullscreenBtn) {
        fullscreenBtn.addEventListener('click', () => {
            const videoContainer = document.querySelector('.video-feed');
            if (videoContainer) {
                if (!document.fullscreenElement) {
                    videoContainer.requestFullscreen().catch(err => {
                        console.error(`Error attempting to enable fullscreen: ${err.message}`);
                    });
                    fullscreenBtn.innerHTML = '<i class="fas fa-compress"></i>';
                } else {
                    document.exitFullscreen();
                    fullscreenBtn.innerHTML = '<i class="fas fa-expand"></i>';
                }
            }
        });
    }

    // Monitoring Controls
    if (startMonitoringBtn) {
        startMonitoringBtn.addEventListener('click', () => {
            isMonitoring = true;
            startMonitoringBtn.classList.add('hidden');
            if (stopMonitoringBtn) stopMonitoringBtn.classList.remove('hidden');
            socket.emit('start_monitoring');
            console.log('Monitoring started');
            
            // Start monitoring timer
            monitoringStartTime = new Date();
            if (monitoringTimer) clearInterval(monitoringTimer);
            monitoringTimer = setInterval(updateMonitoringTime, 1000);
        });
    }

    if (stopMonitoringBtn) {
        stopMonitoringBtn.addEventListener('click', () => {
            isMonitoring = false;
            stopMonitoringBtn.classList.add('hidden');
            if (startMonitoringBtn) startMonitoringBtn.classList.remove('hidden');
            socket.emit('stop_monitoring');
            console.log('Monitoring stopped');
            
            // Stop monitoring timer
            if (monitoringTimer) {
                clearInterval(monitoringTimer);
                monitoringTimer = null;
            }
        });
    }

    // View Toggle
    if (gridViewBtn) {
        gridViewBtn.addEventListener('click', () => {
            if (studentGrid) studentGrid.classList.remove('hidden');
            if (analyticsView) analyticsView.classList.add('hidden');
            gridViewBtn.classList.remove('btn-outline');
            gridViewBtn.classList.add('btn-primary');
            if (analyticsViewBtn) {
                analyticsViewBtn.classList.remove('btn-primary');
                analyticsViewBtn.classList.add('btn-outline');
            }
        });
    }

    if (analyticsViewBtn) {
        analyticsViewBtn.addEventListener('click', () => {
            if (analyticsView) analyticsView.classList.remove('hidden');
            if (studentGrid) studentGrid.classList.add('hidden');
            analyticsViewBtn.classList.remove('btn-outline');
            analyticsViewBtn.classList.add('btn-primary');
            if (gridViewBtn) {
                gridViewBtn.classList.remove('btn-primary');
                gridViewBtn.classList.add('btn-outline');
            }
            updateAnalyticsView();
        });
    }

    // Filtering and Sorting
    if (riskFilter) {
        riskFilter.addEventListener('change', updateAnalyticsView);
    }

    if (sortBy) {
        sortBy.addEventListener('change', updateAnalyticsView);
    }

    if (studentSearch) {
        studentSearch.addEventListener('input', updateStudentList);
    }

    // Create student card
    function createStudentCard(student) {
        const card = document.createElement('div');
        card.className = 'student-card';
        card.id = `student-card-${student.id}`;
        
        const riskLevel = calculateRiskLevel(student);
        
        card.innerHTML = `
            <div class="student-header">
                <div class="student-id">${student.id}</div>
                <div class="risk-badge ${riskLevel.toLowerCase()}">${riskLevel}</div>
            </div>
            <div class="student-stats">
                <div class="student-stat">
                    <div class="student-stat-label">Head Turns</div>
                    <div class="student-stat-value">${student.headTurns || 0}</div>
                </div>
                <div class="student-stat">
                    <div class="student-stat-label">Hand Gestures</div>
                    <div class="student-stat-value">${student.handGestures || 0}</div>
                </div>
                <div class="student-stat">
                    <div class="student-stat-label">Phone Usage</div>
                    <div class="student-stat-value">${student.phoneDetections || 0}</div>
                </div>
            </div>
        `;
        
        // Add click event to show student details
        card.addEventListener('click', () => {
            selectedStudentId = student.id;
            showStudentDetails(student);
            
            // If on dashboard tab, switch to analytics tab
            const analyticsTab = document.querySelector('.nav-item[data-tab="analytics"]');
            if (analyticsTab) {
                analyticsTab.click();
            }
        });
        
        return card;
    }

    // Update analytics view with filtering and sorting
    function updateAnalyticsView() {
        if (!analyticsBody) return;
        
        analyticsBody.innerHTML = '';
        
        // Get filter and sort values
        const riskFilterValue = riskFilter ? riskFilter.value : 'all';
        const sortByValue = sortBy ? sortBy.value : 'id';
        
        // Convert Map to array for filtering and sorting
        let studentsArray = Array.from(studentData.entries()).map(([id, data]) => ({
            id,
            ...data,
            riskLevel: calculateRiskLevel(data)
        }));
        
        // Apply risk filter
        if (riskFilterValue !== 'all') {
            studentsArray = studentsArray.filter(student => 
                student.riskLevel.toLowerCase() === riskFilterValue
            );
        }
        
        // Apply sorting
        studentsArray.sort((a, b) => {
            switch (sortByValue) {
                case 'risk':
                    const riskOrder = { 'High': 3, 'Medium': 2, 'Low': 1 };
                    return riskOrder[b.riskLevel] - riskOrder[a.riskLevel];
                case 'head-turns':
                    return (b.headTurns || 0) - (a.headTurns || 0);
                case 'hand-gestures':
                    return (b.handGestures || 0) - (a.handGestures || 0);
                case 'phone':
                    return (b.phoneDetections || 0) - (a.phoneDetections || 0);
                case 'id':
                default:
                    return a.id.localeCompare(b.id);
            }
        });
        
        // Create table rows
        studentsArray.forEach(student => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${student.id}</td>
                <td>${student.headTurns || 0}</td>
                <td>${student.handGestures || 0}</td>
                <td>${student.phoneDetections || 0}</td>
                <td><span class="risk-badge ${student.riskLevel.toLowerCase()}">${student.riskLevel}</span></td>
                <td><button class="action-btn">View Details</button></td>
            `;
            
            // Add click event to view details
            const viewButton = row.querySelector('.action-btn');
            if (viewButton) {
                viewButton.addEventListener('click', () => {
                    selectedStudentId = student.id;
                    showStudentDetails(student);
                });
            }
            
            analyticsBody.appendChild(row);
        });
    }

    // Update student list in the Students tab
    function updateStudentList() {
        if (!studentList) return;
        
        const searchTerm = studentSearch ? studentSearch.value.toLowerCase() : '';
        
        // Clear current list
        studentList.innerHTML = '';
        
        // If no students, show empty state
        if (studentData.size === 0) {
            studentList.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-users"></i>
                    <p>No students found. Start monitoring to detect students.</p>
                </div>
            `;
            return;
        }
        
        // Filter students by search term
        const filteredStudents = Array.from(studentData.entries())
            .filter(([id, data]) => id.toLowerCase().includes(searchTerm))
            .map(([id, data]) => ({
                id,
                ...data,
                riskLevel: calculateRiskLevel(data)
            }));
        
        // If no matches, show empty state
        if (filteredStudents.length === 0) {
            studentList.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-search"></i>
                    <p>No students match your search.</p>
                </div>
            `;
            return;
        }
        
        // Create student list
        const listElement = document.createElement('div');
        listElement.className = 'student-cards';
        
        filteredStudents.forEach(student => {
            listElement.appendChild(createStudentCard(student));
        });
        
        studentList.appendChild(listElement);
    }

    // Show detailed view of a student
    function showStudentDetails(student) {
        if (!studentDetails) return;
        
        const riskLevel = calculateRiskLevel(student);
        const totalViolations = (student.headTurns || 0) + (student.handGestures || 0) + (student.phoneDetections || 0);
        
        studentDetails.innerHTML = `
            <div class="student-detail-card">
                <div class="student-profile">
                    <div class="student-avatar">
                        <i class="fas fa-user-graduate"></i>
                    </div>
                    <div class="student-name">Student ${student.id}</div>
                    <div class="student-risk risk-badge ${riskLevel.toLowerCase()}">${riskLevel} Risk</div>
                </div>
                <div class="student-data">
                    <div class="student-stats-grid">
                        <div class="student-stat-card">
                            <div class="student-stat-value">${student.headTurns || 0}</div>
                            <div class="student-stat-label">Head Turns</div>
                        </div>
                        <div class="student-stat-card">
                            <div class="student-stat-value">${student.handGestures || 0}</div>
                            <div class="student-stat-label">Hand Gestures</div>
                        </div>
                        <div class="student-stat-card">
                            <div class="student-stat-value">${student.phoneDetections || 0}</div>
                            <div class="student-stat-label">Phone Usage</div>
                        </div>
                        <div class="student-stat-card">
                            <div class="student-stat-value">${totalViolations}</div>
                            <div class="student-stat-label">Total Violations</div>
                        </div>
                    </div>
                    
                    <div class="student-timeline">
                        <h4 class="timeline-title">Activity Timeline</h4>
                        ${generateTimeline(student)}
                    </div>
                </div>
            </div>
        `;
    }

    // Generate timeline for student details
    function generateTimeline(student) {
        // In a real app, we would have a history of events
        // For now, we'll generate some sample events based on the violation counts
        const events = [];
        
        if (student.headTurns > 0) {
            events.push({
                type: 'head-turn',
                icon: '<i class="fas fa-sync-alt"></i>',
                time: '2 minutes ago',
                text: 'Head turn detected'
            });
        }
        
        if (student.handGestures > 0) {
            events.push({
                type: 'hand-gesture',
                icon: '<i class="fas fa-hand-paper"></i>',
                time: '5 minutes ago',
                text: 'Hand gesture detected'
            });
        }
        
        if (student.phoneDetections > 0) {
            events.push({
                type: 'phone',
                icon: '<i class="fas fa-mobile-alt"></i>',
                time: '8 minutes ago',
                text: 'Phone usage detected'
            });
        }
        
        // If no events, show a message
        if (events.length === 0) {
            return `<p>No activity recorded yet.</p>`;
        }
        
        // Generate timeline HTML
        return events.map(event => `
            <div class="timeline-item">
                <div class="timeline-icon">${event.icon}</div>
                <div class="timeline-content">
                    <div class="timeline-time">${event.time}</div>
                    <div class="timeline-text">${event.text}</div>
                </div>
            </div>
        `).join('');
    }

    // Update analytics summary in the Analytics tab
    function updateAnalyticsSummary() {
        if (totalViolations) {
            const violations = totalHeadTurns + totalHandGestures + totalPhoneDetections;
            totalViolations.textContent = violations;
        }
        
        if (totalStudents) {
            totalStudents.textContent = studentData.size;
        }
        
        // If a student is selected, show their details
        if (selectedStudentId && studentData.has(selectedStudentId)) {
            showStudentDetails(studentData.get(selectedStudentId));
        }
    }

    // Update monitoring time display
    function updateMonitoringTime() {
        if (!monitoringTime || !monitoringStartTime) return;
        
        const now = new Date();
        const diff = now - monitoringStartTime;
        
        // Convert to hours, minutes, seconds
        const hours = Math.floor(diff / 3600000);
        const minutes = Math.floor((diff % 3600000) / 60000);
        const seconds = Math.floor((diff % 60000) / 1000);
        
        // Format as HH:MM:SS
        monitoringTime.textContent = [
            hours.toString().padStart(2, '0'),
            minutes.toString().padStart(2, '0'),
            seconds.toString().padStart(2, '0')
        ].join(':');
    }

    // Calculate risk level
    function calculateRiskLevel(student) {
        const headTurns = student.headTurns || 0;
        const handGestures = student.handGestures || 0;
        const phoneDetections = student.phoneDetections || 0;
        
        const totalViolations = headTurns + handGestures + (phoneDetections * 2);
        
        if (totalViolations >= 5 || phoneDetections >= 1) {
            return 'High';
        } else if (totalViolations >= 2) {
            return 'Medium';
        } else {
            return 'Low';
        }
    }

    // Socket.IO event handlers
    socket.on('connect', () => {
        console.log('Connected to server');
    });

    socket.on('detection_result', (data) => {
        console.log('Detection result received:', data);
        
        if (!data.student_id) return;
        
        // Get or create student data
        if (!studentData.has(data.student_id)) {
            studentData.set(data.student_id, {
                id: data.student_id,
                headTurns: 0,
                handGestures: 0,
                phoneDetections: 0,
                events: []
            });
            
            // Update active students count
            if (activeStudents) {
                activeStudents.textContent = studentData.size;
            }
            
            // Update student list
            updateStudentList();
        }
        
        const student = studentData.get(data.student_id);
        let updated = false;
        
        // Check for head turn
        if (data.head_angle && Math.abs(data.head_angle) > 30) {
            student.headTurns++;
            totalHeadTurns++;
            if (headTurnCount) headTurnCount.textContent = totalHeadTurns;
            
            // Add event to student history
            student.events.push({
                type: 'head-turn',
                time: new Date(),
                angle: data.head_angle
            });
            
            updated = true;
        }
        
        // Check for hand gesture
        if (data.hands_detected) {
            student.handGestures++;
            totalHandGestures++;
            if (handGestureCount) handGestureCount.textContent = totalHandGestures;
            
            // Add event to student history
            student.events.push({
                type: 'hand-gesture',
                time: new Date()
            });
            
            updated = true;
        }
        
        // Check for phone detection
        if (data.detections && data.detections.some(d => d.class === 'cell phone')) {
            student.phoneDetections++;
            totalPhoneDetections++;
            if (phoneCount) phoneCount.textContent = totalPhoneDetections;
            
            // Add event to student history
            student.events.push({
                type: 'phone',
                time: new Date(),
                confidence: data.detections.find(d => d.class === 'cell phone').confidence
            });
            
            updated = true;
        }
        
        // Update total alert count
        if (alertCount) {
            alertCount.textContent = totalHeadTurns + totalHandGestures + totalPhoneDetections;
        }
        
        // Update student card if it exists, or create a new one
        const existingCard = document.getElementById(`student-card-${data.student_id}`);
        if (existingCard && studentGrid) {
            studentGrid.replaceChild(createStudentCard(student), existingCard);
        } else if (studentGrid) {
            studentGrid.appendChild(createStudentCard(student));
        }
        
        // Update analytics view if visible
        if (analyticsView && !analyticsView.classList.contains('hidden')) {
            updateAnalyticsView();
        }
        
        // Update student details if this student is selected
        if (updated && selectedStudentId === data.student_id) {
            showStudentDetails(student);
        }
        
        // Update analytics summary
        updateAnalyticsSummary();
    });

    // Handle video feed errors
    if (mainVideo) {
        mainVideo.onerror = function() {
            console.error('Video feed error');
            // Try to reload the video feed
            setTimeout(() => {
                mainVideo.src = mainVideo.src.split('?')[0] + '?' + new Date().getTime();
            }, 5000);
        };
    }
});