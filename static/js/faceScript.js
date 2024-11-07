let isFetching = false;
let currentUserName = ""; // Variable to store the current user's name

function fetchData() {
    if (isFetching) return; // Prevent overlap
    isFetching = true;

    fetch('/user/get_detected_info')
        .then(response => response.json())
        .then(data => {
            // Update UI with detected info
            document.getElementById('detected-name').innerText = data.name || "Unknown";
            document.getElementById('detected-time').innerText = data.datetime || "";
            const grade = data.grade_level || "";
            const section = data.section || "";
            document.getElementById('detected-grade-section').innerText = `${grade} ${section}`;

            // Set the image source if the user is recognized
            const userPicture = document.getElementById('userPicture');
            if (data.name && data.picture_path) {
                userPicture.src = `/user/datasets/${data.picture_path}`; // Set the source based on the path
                userPicture.style.display = 'block'; // Show the image
            } else {
                userPicture.style.display = 'none'; // Hide the image if unknown
            }

            // Store the current user's name for attendance fetching
            currentUserName = data.name || "";

            // Fetch user data only if a user is detected
            if (currentUserName) {
                return Promise.all([
                    fetchAllUsersData(currentUserName),
                    fetchAttendance(currentUserName) // Fetch attendance data
                ]);
            }
        })
        .catch(error => console.error('Error fetching detected info:', error))
        .finally(() => {
            isFetching = false; // Reset fetching status
        });
}

// Fetch data every second
setInterval(fetchData, 1000);

function fetchAllUsersData(name) {
    if (!name) {
        console.error("Name is required to fetch user data.");
        return Promise.resolve(); // Return a resolved promise if name is not provided
    }

    let url = `/user/get_all_users_data/${encodeURIComponent(name)}`;

    return fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok: ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            const userTable = document.getElementById('userTable');

            // Clear existing rows (except for header)
            userTable.innerHTML = `
                <tr class="bg-gray-100">
                    <th class="text-gray-600 border border-gray-300 p-1">#</th>
                    <th class="text-gray-600 border border-gray-300 p-1">Name</th>
                    <th class="text-gray-600 border border-gray-300 p-1">Grade</th>
                    <th class="text-gray-600 border border-gray-300 p-1">Section</th>
                    <th class="text-gray-600 border border-gray-300 p-1">Entry Datetime</th>
                    <th class="text-gray-600 border border-gray-300 p-1">Period</th>
                </tr>
            `;

            // Populate table with fetched data
            data.forEach((user, index) => {
                userTable.innerHTML += `
                    <tr>
                        <td class="text-gray-600 border-b border-gray-300 p-1 text-center">${index + 1}</td>
                        <td class="text-gray-600 border-b border-gray-300 p-1 text-center">${user.name}</td>
                        <td class="text-gray-600 border-b border-gray-300 p-1 text-center">${user.grade_level}</td>
                        <td class="text-gray-600 border-b border-gray-300 p-1 text-center">${user.section}</td>
                        <td class="text-gray-600 border-b border-gray-300 p-1 text-center">${user.entry_datetime}</td>
                        <td class="text-gray-600 border-b border-gray-300 p-1 text-center">${user.period}</td>
                    </tr>
                `;
            });
        })
        .catch(error => console.error('Error fetching user data:', error));
}

function fetchAttendance(name) {
    if (!name) {
        console.error("Name is required to fetch attendance data.");
        return;
    }

    fetch(`/user/attendance/${encodeURIComponent(name)}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok: ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            // Update UI with attendance data
            document.getElementById('attendance-total').innerText = data.total_attendance || "0";
            document.getElementById('attendance-weekly').innerText = data.weekly_attendance || "0";
        })
        .catch(error => console.error('Error fetching attendance data:', error));
}

// Call this function when the page loads
document.addEventListener('DOMContentLoaded', () => {
    fetchData(); // Initial fetch
});

// Set an interval for fetching attendance using the current user's name
setInterval(() => {
    fetchAttendance(currentUserName); // Pass the current user's name
}, 1000);
