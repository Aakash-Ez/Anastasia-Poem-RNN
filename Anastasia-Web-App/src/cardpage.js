import React from 'react';
import ReactDOM from 'react-dom';
import Navbar from './Navbar'
function Body(){

}
class Page extends React.Component{
    render(){
        return(
            <div className="Page">
                <Navbar/>
                <Body/>
            </div>
        )
    }
}
ReactDOM.render(
    <Page/>,
  document.getElementById('root')
);