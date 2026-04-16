"use client";
import React from 'react';

function ThemeSwitcher() {
	function changeThemeMode(event) {
		event.preventDefault();
		document.body.classList.toggle('dark-theme');
		event.currentTarget.classList.toggle('dark');
	}

	return (
		<div className="fixed_floating_icons">
			<a href="#0" className="item dark_light_btn" onClick={changeThemeMode}>
				<span className="icon"> <i className="fas fa-sun"></i> </span>
				<span className="icon dark_icon"> <i className="fas fa-moon"></i> </span>
			</a>
		</div>
	)
}

export default ThemeSwitcher