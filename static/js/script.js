function showPage(pageId) {
    document.querySelectorAll('.page').forEach(page => {
        page.classList.add('hidden');
    });
    document.getElementById(pageId).classList.remove('hidden');

    let newTitle = '';
    switch(pageId) {
        case 'dashboard':
            newTitle = 'Admin | Dashboard';
            break;
        case 'attendance':
            newTitle = 'Admin | Attendance Records';
            break;
        case 'activity':
            newTitle = 'Admin | Activity Logs';
            break;
        case 'backup':
            newTitle = 'Admin | Database Backup';
            break;
        default:
            newTitle = 'Admin | Dashboard';
    }
    document.title = newTitle;

    document.querySelectorAll('.w-64 ul li a').forEach(link => {
        link.classList.remove('bg-sky-600');
    });
    document.getElementById('link-' + pageId).classList.add('bg-sky-600');
}

showPage('dashboard');

function updateDateTime() {
    const now = new Date();
    const optionsDate = { year: 'numeric', month: 'long', day: 'numeric' };
    const optionsTime = { 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit', 
        hour12: true 
    };

    const formattedDate = now.toLocaleDateString(undefined, optionsDate);
    const formattedTime = now.toLocaleTimeString(undefined, optionsTime);

    document.getElementById('date').textContent = formattedDate;
    document.getElementById('time').textContent = formattedTime;
}

setInterval(updateDateTime, 1000);

updateDateTime();

let allUsers = []; // Store all fetched users

// Customize sections for each grade
const gradeSections = {
    "Grade 7": ["Section 1", "Section 2"],
    "Grade 8": ["Rizal", "Aguinaldo",],
    "Grade 9": ["Masigasig", "Matapat", "Maligalig", "SPJ"],
    "Grade 10": ["Love", "Faith", "Hope", "SPJ"],
    "Grade 11": ["Humss", "TVL-A", "TVL-B", "HE"],
    "Grade 12": ["Humss", "TVL I-A", "TVL II-B", "HE"]
};

function fetchAllUsers() {
    fetch('/admin/get_all_users')
        .then(response => response.json())
        .then(data => {
            allUsers = data; // Store all user data
            renderUsers(allUsers); // Render users
            updateSections(); // Populate sections based on initial selection
        })
        .catch(error => console.error('Error fetching user data:', error));
}

function renderUsers(users) {
    const userTableBody = document.getElementById('attendanceTable').getElementsByTagName('tbody')[0];
    userTableBody.innerHTML = ''; // Clear existing rows

    if (!Array.isArray(users) || users.length === 0) {
        const row = userTableBody.insertRow();
        const cell = row.insertCell(0);
        cell.colSpan = 6;
        cell.innerText = 'No attendance records available.';
        cell.style.textAlign = 'center';
        return;
    }

    users.forEach(user => {
        const row = userTableBody.insertRow();
        row.insertCell(0).innerText = user.id;
        row.insertCell(1).innerText = user.name;
        row.insertCell(2).innerText = user.status || "Present"; // Default status if missing
        row.insertCell(3).innerText = user.grade_level;
        row.insertCell(4).innerText = user.section;
        row.insertCell(5).innerText = user.user_type;
        row.insertCell(6).innerText = user.time_in;
        row.insertCell(7).innerText = user.time_out;
    });
}

function updateSections() {
    const gradeSelect = document.getElementById('gradeSelect');
    const sectionSelect = document.getElementById('sectionSelect');
    const selectedGrade = gradeSelect.value;

    // Clear existing sections
    sectionSelect.innerHTML = '<option selected>Section</option>';

    // Populate sections based on the selected grade
    if (selectedGrade && gradeSections[selectedGrade]) {
        gradeSections[selectedGrade].forEach(section => {
            const option = document.createElement('option');
            option.value = section;
            option.textContent = section.replace('_', ' ').replace('section ', 'Section '); // Format nicely
            sectionSelect.appendChild(option);
        });
    }
}

function filterUsers() {
    const searchInput = document.getElementById('searchInputUsers').value.toLowerCase();
    const selectedGrade = document.getElementById('gradeSelect').value;
    const selectedSection = document.getElementById('sectionSelect').value;

    const filteredUsers = allUsers.filter(user => {
        const matchesSearch = 
            (user.name && user.name.toLowerCase().includes(searchInput)) ||
            (user.grade_level && user.grade_level.toLowerCase().includes(searchInput)) ||
            (user.section && user.section.toLowerCase().includes(searchInput)) ||
            (user.entry_datetime && user.entry_datetime.toLowerCase().includes(searchInput));

        const matchesGrade = selectedGrade === "Grade Level" || user.grade_level === selectedGrade;
        const matchesSection = selectedSection === "Section" || user.section === selectedSection;

        return matchesSearch && matchesGrade && matchesSection;
    });

    renderUsers(filteredUsers);
}

function sortUsers() {
    const sortBy = document.getElementById('sortSelectUsers').value;
    let sortedUsers;

    if (sortBy === 'date') {
        sortedUsers = [...allUsers].sort((a, b) => new Date(b.entry_datetime) - new Date(a.entry_datetime));
    } else if (sortBy === 'name') {
        sortedUsers = [...allUsers].sort((a, b) => a.name.localeCompare(b.name));
    } else if (sortBy === 'id') {
        sortedUsers = [...allUsers].sort((a, b) => a.id - b.id);
    } else if (sortBy === 'student') {
        sortedUsers = [...allUsers].filter(a => a.user_type === "Student");
    } else if (sortBy === 'teacher') {
        sortedUsers = [...allUsers].filter(a => a.user_type === "Teacher");
    } else if (sortBy === 'staff') {
        sortedUsers = [...allUsers].filter(a => a.user_type === "Staff");
    } else {
        return; 
    }

    renderUsers(sortedUsers); 
}

function exportToExcelAttendance() {
    const userTable = document.getElementById('attendanceTable');
    const rows = Array.from(userTable.rows).slice(1); // Exclude header row

    const formattedLogs = rows.map(row => ({
        id: row.cells[0].innerText,
        name: row.cells[1].innerText,
        status: row.cells[2].innerText,
        grade: row.cells[3].innerText,
        section: row.cells[4].innerText,
        user_type: row.cells[5].innerText,
        time_in: row.cells[6].innerText,
        time_out: row.cells[7].innerText,
    }));

    const ws = XLSX.utils.json_to_sheet(formattedLogs);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "Attendance Records");

    XLSX.writeFile(wb, "attendance_records.xlsx");
}

// Call the function when the page loads
document.addEventListener('DOMContentLoaded', fetchAllUsers);

let registeredUsers = []; // Store the registered users

// Fetch registered users data
function fetchRegisteredUsers() {
    fetch('/admin/get_registered_users') // Endpoint for registered users
        .then(response => response.json())
        .then(data => {
            registeredUsers = data; // Store the fetched registered users
            renderRegisteredUsers(registeredUsers); // Render the users in the table
        })
        .catch(error => console.error('Error fetching registered users data:', error));
}

// Call the function when the page loads
document.addEventListener('DOMContentLoaded', fetchRegisteredUsers);

// Render the registered users data in the table
function renderRegisteredUsers(users) {
    const tbody = document.getElementById('registeredUsersTable').getElementsByTagName('tbody')[0];
    tbody.innerHTML = ''; // Clear existing rows

    if (!Array.isArray(users) || users.length === 0) {
        const row = tbody.insertRow();
        const cell = row.insertCell(0);
        cell.colSpan = 8;
        cell.innerText = 'No registered users available.';
        cell.style.textAlign = 'center';
        return;
    }

    users.forEach(user => {
        const row = tbody.insertRow();
        row.insertCell(0).innerText = user.id;
        row.insertCell(1).innerText = user.name;
        row.insertCell(2).innerText = user.grade_level;
        row.insertCell(3).innerText = user.section;
        row.insertCell(4).innerText = user.user_type;
        row.insertCell(5).innerText = user.total_attendance;
        row.insertCell(6).innerText = user.weekly_attendance;
        row.insertCell(7).innerText = user.week;
        row.insertCell(8).innerText = user.date_created;
    });
}

// Filter registered users based on search, grade, and section
function filterRegisteredUsers() {
    const searchInput = document.getElementById('searchInputRegisteredUsers').value.toLowerCase();
    const selectedGrade = document.getElementById('gradeSelectRegisteredUsers').value;
    const selectedSection = document.getElementById('sectionSelectRegisteredUsers').value;

    const filteredUsers = registeredUsers.filter(user => {
        const matchesSearch = 
            (user.name && user.name.toLowerCase().includes(searchInput)) ||
            (user.grade_level && user.grade_level.toLowerCase().includes(searchInput)) ||
            (user.section && user.section.toLowerCase().includes(searchInput)) ||
            (user.date_created && user.date_created.toLowerCase().includes(searchInput));

        const matchesGrade = selectedGrade === "Grade Level" || user.grade_level === selectedGrade;
        const matchesSection = selectedSection === "Section" || user.section === selectedSection;

        return matchesSearch && matchesGrade && matchesSection;
    });

    renderRegisteredUsers(filteredUsers); // Render the filtered users
}

// Sort registered users based on selected criteria
function sortRegisteredUsers() {
    const sortBy = document.getElementById('sortSelectRegisteredUsers').value;
    let sortedUsers;

    if (sortBy === 'date') {
        sortedUsers = [...registeredUsers].sort((a, b) => new Date(b.date_created) - new Date(a.date_created));
    } else if (sortBy === 'name') {
        sortedUsers = [...registeredUsers].sort((a, b) => a.name.localeCompare(b.name));
    } else if (sortBy === 'id') {
        sortedUsers = [...registeredUsers].sort((a, b) => a.id - b.id);
    } else if (sortBy === 'student') {
        sortedUsers = [...registeredUsers].filter(a => a.user_type === "Student");
    } else if (sortBy === 'teacher') {
        sortedUsers = [...registeredUsers].filter(a => a.user_type === "Teacher");
    } else if (sortBy === 'staff') {
        sortedUsers = [...registeredUsers].filter(a => a.user_type === "Staff");
    } else {
        return;  
    }    

    renderRegisteredUsers(sortedUsers); // Render the sorted users
}

// Excel export for registered users
function exportToExcelRegisteredUsers() {
    const formattedData = registeredUsers.map(user => ({
        id: user.id,
        name: user.name,
        grade_level: user.grade_level,
        section: user.section,
        user_type: user.user_type,
        total_attendance: user.total_attendance,
        weekly_attendance: user.weekly_attendance,
        week: user.week,
        date_created: user.date_created
    }));

    const ws = XLSX.utils.json_to_sheet(formattedData);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "Registered Users");
    XLSX.writeFile(wb, "registered_users.xlsx");
}

// Customize sections for each grade for registered users
const gradeSectionsForRegisteredUsers = {
    "Grade 7": ["Section 1", "Section 2"],
    "Grade 8": ["Rizal", "Aguinaldo",],
    "Grade 9": ["Masigasig", "Matapat", "Maligalig", "SPJ"],
    "Grade 10": ["Love", "Faith", "Hope", "SPJ"],
    "Grade 11": ["Humss", "TVL-A", "TVL-B", "HE"],
    "Grade 12": ["Humss", "TVL I-A", "TVL II-B", "HE"]
};

// Update sections based on the selected grade for registered users
function updateSectionsRegisteredUsers() {
    const gradeSelect = document.getElementById('gradeSelectRegisteredUsers');
    const sectionSelect = document.getElementById('sectionSelectRegisteredUsers');
    const selectedGrade = gradeSelect.value;

    // Clear existing sections
    sectionSelect.innerHTML = '<option selected>Section</option>';

    // Populate sections based on the selected grade
    if (selectedGrade && gradeSectionsForRegisteredUsers[selectedGrade]) {
        gradeSectionsForRegisteredUsers[selectedGrade].forEach(section => {
            const option = document.createElement('option');
            option.value = section;
            option.textContent = section.replace('_', ' ').replace('section ', 'Section '); // Format nicely
            sectionSelect.appendChild(option);
        });
    }
}

// Call the function to update sections when the page loads
document.addEventListener('DOMContentLoaded', updateSectionsRegisteredUsers);


let allLogs = []; // Store the fetched logs

function fetchAllLogs() {
    fetch('/admin/get_activity_logs')
        .then(response => {
            if (!response.ok) {
                return response.json().then(errorData => {
                    throw new Error(errorData.error || 'Failed to fetch logs');
                });
            }
            return response.json();
        })
        .then(data => {
            allLogs = data; // Store all logs
            renderLogs(allLogs); // Render logs

        })
        .catch(error => console.error('Error fetching logs data:', error));
}

function renderLogs(logs) {
    const logsTable = document.getElementById('activityLogsTable').getElementsByTagName('tbody')[0];
    logsTable.innerHTML = ''; // Clear existing rows

    if (!Array.isArray(logs) || logs.length === 0) {
        const row = logsTable.insertRow();
        const cell = row.insertCell(0);
        cell.colSpan = 4;
        cell.innerText = 'No activity logs available.';
        cell.style.textAlign = 'center';
        return;
    }

    logs.forEach(log => {
        const row = logsTable.insertRow();
        row.insertCell(0).innerText = log.id;
        row.insertCell(1).innerText = log.name;
        row.insertCell(2).innerText = log.activity;
        row.insertCell(3).innerText = log.date_time;
    });
}

function filterLogs() {
    const searchInput = document.getElementById('searchInputLogs').value.toLowerCase();
    const filteredLogs = allLogs.filter(log => {
        return (
            log.name.toLowerCase().includes(searchInput) ||
            log.activity.toLowerCase().includes(searchInput) ||
            log.date_time.toLowerCase().includes(searchInput)
        );
    });
    renderLogs(filteredLogs); // Render the filtered logs
}

function sortLogs() {
    const sortBy = document.getElementById('sortSelectLogs').value;
    let sortedLogs;

    if (sortBy === 'date') {
        sortedLogs = [...allLogs].sort((a, b) => new Date(b.date_time) - new Date(a.date_time));
    } else if (sortBy === 'name') {
        sortedLogs = [...allLogs].sort((a, b) => a.name.localeCompare(b.name));
    } else if (sortBy === 'id') {
        sortedLogs = [...allLogs].sort((a, b) => a.id - b.id); // Sort by ID
    } else {
        return; // No sorting if the default option is selected
    }

    renderLogs(sortedLogs); // Render the sorted logs
}

function exportToExcel() {
    // Check if allLogs has data
    console.log(allLogs); // For debugging

    // Transform the logs to ensure the order is ID, Name, Activity, Date & Time
    const formattedLogs = allLogs.map(log => ({
        id: log.id,
        name: log.name,
        activity: log.activity,
        date_time: log.date_time
    }));

    const ws = XLSX.utils.json_to_sheet(formattedLogs);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "Activity Logs");

    // Create an XLSX file and trigger the download
    XLSX.writeFile(wb, "activity_logs.xlsx");
}

// Call the function when the page loads
document.addEventListener('DOMContentLoaded', fetchAllLogs);

function confirmBackup() {
    const form = document.createElement('form');
    form.method = 'POST';
    form.action = '/admin/backup';
    document.body.appendChild(form);
    form.submit(); // Trigger the download
    backupDatabase.close(); // Close the modal
}

// Function to fetch the count of unique students, teachers, staff, and total users
async function fetchAllUserCounts() {
    try {
        // Fetch data from the new endpoint
        const response = await fetch('/admin/count_all_users');
        const data = await response.json();

        // Update the HTML elements with the respective counts
        document.getElementById('unique-students-count').textContent = data.students_count;
        document.getElementById('unique-teachers-count').textContent = data.teachers_count;
        document.getElementById('unique-staffs-count').textContent = data.staffs_count;
        document.getElementById('total-users-count').textContent = data.total_users_count;

    } catch (error) {
        console.error('Error fetching user counts:', error);
        // Show an error message in case of failure
        document.getElementById('unique-students-count').textContent = 'Error';
        document.getElementById('unique-teachers-count').textContent = 'Error';
        document.getElementById('unique-staffs-count').textContent = 'Error';
        document.getElementById('total-users-count').textContent = 'Error';
    }
}

// Function to fetch total and daily attendance counts
async function fetchAttendanceCounts() {
    try {
        // Fetch data from the count_attendance endpoint
        const response = await fetch('/admin/count_attendance');
        const data = await response.json();

        // Update the HTML elements with the respective counts
        document.getElementById('total-attendance-count').textContent = data.total_attendance;
        document.getElementById('daily-attendance-count').textContent = data.daily_attendance;

    } catch (error) {
        console.error('Error fetching attendance counts:', error);
        // Show an error message in case of failure
        document.getElementById('total-attendance-count').textContent = 'Error';
        document.getElementById('daily-attendance-count').textContent = 'Error';
    }
}

// Function to fetch monthly attendance data from the Flask backend
async function fetchMonthlyAttendanceData() {
    try {
        // Fetch data from the backend
        const response = await fetch('/admin/monthly_attendance');
        const data = await response.json();

        // Prepare data for the chart
        const labels = data.labels;  // Full month names
        const attendanceCounts = data.attendance_counts;  // The attendance counts

        // Generate a random color for each bar
        function getRandomColor() {
            const letters = '0123456789ABCDEF';
            let color = '#';
            for (let i = 0; i < 6; i++) {
                color += letters[Math.floor(Math.random() * 16)];
            }
            return color;
        }

        // Generate random colors for the bars
        const colors = labels.map(() => getRandomColor());

        // Get the context of the canvas element
        const ctx = document.getElementById('barChart').getContext('2d');

        // Create a new Chart.js bar chart
        const chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,  // X-axis labels (month names)
                datasets: [{
                    label: 'Attendance Count',
                    data: attendanceCounts,  // Y-axis data (attendance counts)
                    backgroundColor: colors,  // Random bar colors
                    borderColor: colors,  // Border color for each bar
                    borderWidth: 1  // Border width for the bars
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true  // Start Y-axis at 0
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error fetching monthly attendance:', error);
    }
}

window.onload = function() {
    fetchAllUserCounts();
    fetchAttendanceCounts();
    fetchMonthlyAttendanceData();
};

function printAttendance() {
    const table = document.getElementById('attendanceTable');
    const printWindow = window.open('', '', 'height=600,width=800');
    
    printWindow.document.write('<html><head><title>Print Attendance</title>');
    printWindow.document.write('<link rel="stylesheet" href="{{ url_for(\'static\', filename=\'css/output.css\') }}">'); // Include your CSS for styling
    printWindow.document.write(`
        <style>
            body {
                font-family: Arial, sans-serif;
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                border: 1px solid black;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            .logo {
                width: 150px; /* Adjust the width as necessary */
                margin-bottom: 20px; /* Space below the logo */
            }
        </style>
    `);
    printWindow.document.write('</head><body>');
    
    // Add the logo
    printWindow.document.write('<img src="{{ url_for(\'static\', filename=\'img/school-logo.png\') }}" class="logo" alt="School Logo" style="display: block; margin-left: auto; margin-right: auto;">');
    
    printWindow.document.write('<h2>Attendance Records</h2>');
    printWindow.document.write(table.outerHTML);
    printWindow.document.write('</body></html>');
    
    printWindow.document.close();
    printWindow.print();
}

function printRegistered() {
    const table = document.getElementById('registeredUsersTable');
    const printWindow = window.open('', '', 'height=600,width=800');
    
    printWindow.document.write('<html><head><title>Print Registered Users</title>');
    printWindow.document.write('<link rel="stylesheet" href="{{ url_for(\'static\', filename=\'css/output.css\') }}">'); // Include your CSS for styling
    printWindow.document.write(`
        <style>
            body {
                font-family: Arial, sans-serif;
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                border: 1px solid black;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            .logo {
                width: 150px; /* Adjust the width as necessary */
                margin-bottom: 20px; /* Space below the logo */
            }
        </style>
    `);
    printWindow.document.write('</head><body>');
    
    // Add the logo
    printWindow.document.write('<img src="{{ url_for(\'static\', filename=\'img/school-logo.png\') }}" class="logo" alt="School Logo" style="display: block; margin-left: auto; margin-right: auto;">');
    
    printWindow.document.write('<h2>Registered Users</h2>');
    printWindow.document.write(table.outerHTML);
    printWindow.document.write('</body></html>');
    
    printWindow.document.close();
    printWindow.print();
}


function printLogs() {
    const table = document.getElementById('activityLogsTable');
    const printWindow = window.open('', '', 'height=600,width=800');
    
    printWindow.document.write('<html><head><title>Print Activity Logs</title>');
    printWindow.document.write('<link rel="stylesheet" href="{{ url_for(\'static\', filename=\'css/output.css\') }}">'); // Include your CSS for styling
    printWindow.document.write(`
        <style>
            body {
                font-family: Arial, sans-serif;
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                border: 1px solid black;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            .logo {
                width: 150px; /* Adjust the width as necessary */
                margin-bottom: 20px; /* Space below the logo */
            }
        </style>
    `);
    printWindow.document.write('</head><body>');
    
    // Add the logo
    printWindow.document.write('<img src="{{ url_for(\'static\', filename=\'img/school-logo.png\') }}" class="logo" alt="School Logo" style="display: block; margin-left: auto; margin-right: auto;">');
    
    printWindow.document.write('<h2>Activity Logs</h2>');
    printWindow.document.write(table.outerHTML);
    printWindow.document.write('</body></html>');
    
    printWindow.document.close();
    printWindow.print();
}
