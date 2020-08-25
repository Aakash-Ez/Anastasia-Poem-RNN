import React from 'react';
function Navbar(props){
    return(
        <div className="navbar">
            <div className="title nav-item" onClick={() => props.click(false)}>
                Anastasia
            </div>
            <div className="endnav nav-item" onClick={() => props.click(true)}>
                About
            </div>
        </div>
    )
}
export default Navbar;