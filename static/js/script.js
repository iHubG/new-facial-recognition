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
    "grade_7": ["section_1", "section_2", "section_3"],
    "grade_8": ["section_1", "section_4", "section_5"],
    "grade_9": ["section_2", "section_6"],
    "grade_10": ["section_3", "section_7"],
    "grade_11": ["section_1", "section_2"],
    "grade_12": ["section_1", "section_2"]
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
        row.insertCell(5).innerText = user.entry_datetime; // Adjust if needed
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
    } else {
        return; // No sorting if the default option is selected
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
        date_time: row.cells[5].innerText,
    }));

    const ws = XLSX.utils.json_to_sheet(formattedLogs);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "Attendance Records");

    XLSX.writeFile(wb, "attendance_records.xlsx");
}

// Call the function when the page loads
document.addEventListener('DOMContentLoaded', fetchAllUsers);


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

// Function to fetch the count of unique students
async function fetchUniqueStudentCount() {
    try {
    const response = await fetch('/admin/count_all_users'); // Updated URL to match the new endpoint
    const data = await response.json();
    document.getElementById('unique-student-count').textContent = data.unique_users_count;
    } catch (error) {
    console.error('Error fetching student count:', error);
    document.getElementById('unique-student-count').textContent = 'Error';
    }
}

// Call the function to fetch the count when the page loads
window.onload = fetchUniqueStudentCount;