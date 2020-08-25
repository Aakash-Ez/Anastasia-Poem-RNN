import React, { useState } from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import Navbar from './Navbar'
import {animated, useSpring} from 'react-spring';
class CardDisplay extends React.Component{
    constructor(props){
        super(props);
        this.state={
            clicked:[false,false,false,false,false],
            cardView:false
        }
        this.handleClick=this.handleClick.bind(this);
        this.selectPage=this.selectPage.bind(this);
        this.card = "";
    }
    handleClick(index){
        var click;
        if(this.state.clicked[index]){
            click = this.state.clicked;
        }else{
            click = new Array(5).fill(false);
        }
        click[index] = !click[index];
        console.log(click);
        this.setState({clicked:click});
    }
    selectPage(src,classname,e){
        if(src === "close"){
            if(e.getAttribute("class") === "overlay")
            {
                this.setState({cardView:false});
                this.card="";    
            }
        }else{
            this.card= <Overlay class={classname} srcImge={src} click={this.selectPage}/>;
            this.setState({cardView:true});
        }
    }
    render(){
        return(
        <div className="card-disp">
            <Card value ={0} imgrsc="./Images/img0.png" cardClass="fir bord" selectPage={this.selectPage} triggered={this.state.clicked[0]} click={this.handleClick} src={require('./Images/img0.png')}/>
            <Card value ={1} imgrsc="./Images/img1.png" cardClass="sec bord" selectPage={this.selectPage} triggered={this.state.clicked[1]} click={this.handleClick} src={require("./Images/img1.png")}/>
            <Card value ={2} imgrsc="./Images/img2.png" cardClass="thir bord" selectPage={this.selectPage} triggered={this.state.clicked[2]} click={this.handleClick} src={require("./Images/img2.png")}/>
            <Card value ={3} imgrsc="./Images/img3.png" cardClass="four bord" selectPage={this.selectPage} triggered={this.state.clicked[3]} click={this.handleClick} src={require("./Images/img3.png")}/>
            <Card value ={4} imgrsc="./Images/img4.png" cardClass="fif bord" selectPage={this.selectPage} triggered={this.state.clicked[4]} click={this.handleClick} src={require("./Images/img4.png")}/>
            {this.card}
        </div>
        )
    }
}
class Overlay extends React.Component{
    constructor(props){
        super(props);
        this.state={
            height:8,
            width:4,
            numLines:14,
            sonnet:"",
            sonnetoutput:""
        }
        this.changeInput = this.changeInput.bind(this);
        this.submitInput = this.submitInput.bind(this);
        this.createSonnetDiv = this.createSonnetDiv.bind(this);
    }
    submitInput(){
        console.log("hello server");
        var sonnet = {output:"Generating Sonnet"};
        var xhttp = new XMLHttpRequest();
        this.setState({sonnetoutput:sonnet});
        xhttp.onloadstart =() => {
            this.setState({sonnetoutput:sonnet.output});
        }
        xhttp.onloadend = () => {
            sonnet = xhttp.responseText;
            this.setState({sonnetoutput:JSON.parse(sonnet).output});
        }
        xhttp.open("POST","https://young-brook-19650.herokuapp.com/predict",true);
        xhttp.setRequestHeader("Content-type", "application/json");
        xhttp.send(JSON.stringify({sonnet:this.state.sonnet,numLines:this.state.numLines}));
        console.log(JSON.stringify({sonnet:this.state.sonnet,numLines:this.state.numLines}))
        console.log(sonnet);
    }

    changeTextAreaInput(e){
        this.setState({sonnet:e.value})
    }
    changeInput(e){
        if(e.name === "height"){
            this.setState({height:e.value});
        }else if(e.name === "width"){
            this.setState({width:e.value});
        }else{
            this.setState({numLines:e.value});
        }
    }
    createSonnetDiv(str){
        var sonnet_string = str.split('\n');
        var parentDiv = new Array(14);
        for (let i = 0; i < sonnet_string.length; i++) {
            sonnet_string[i] = sonnet_string[i].charAt(0).toUpperCase() + sonnet_string[i].slice(1);
            parentDiv[i] = <div className="sonnet-line">
                {sonnet_string[i]}
            </div>            
        }
        return parentDiv;
    }
    render(){
        var classname = "poem-content "+this.props.class;
        return(
            <div className="overlay" onClick={(e)=>this.props.click("close","",e.target)}>
            <div className="card-dets" >
                <div className="card-view">
                    <img src={this.props.srcImge} alt=""/>
                    <div className={classname}>
                        <p>{this.createSonnetDiv(this.state.sonnetoutput)}
                        </p>
                    </div>
                     <div className="height">
                        {this.state.height}
                     </div>
                     <div className="width">
                        {this.state.width}
                     </div>
                     <div className="vert dimImg">
                         <img src={require("./Images/vert.png")} alt=""/>
                     </div>
                     <div className="horizontal dimImg">
                         <img src={require("./Images/horizontal.png")} alt=""/>
                     </div>
                </div>
                <div className="card-form">
                    <div className="header">
                        Generate Your Sonnet
                    </div>
                    <form>
                        Enter First 50 characters for poem <br/>
                        <textarea rows="10" cols="55" onChange={(e)=>this.changeTextAreaInput(e.target)}></textarea><br/>
                        Height: <input className="dims" type="text" name="height" defaultValue="8" onChange={(e)=>this.changeInput(e.target)}/>
                        Width: <input className="dims" type="text" name="width" defaultValue="4" onChange={(e)=>this.changeInput(e.target)}/><br/>
                        Enter Number of lines requried: <input type="text" name="num" defaultValue="14" className="dims" onChange={(e)=>this.changeInput(e.target)}/> (Max Lines 14) <br/>
                        <button type="button" name="sub" onClick={()=>this.submitInput()} className="btn">Generate Sonnet</button>
                    </form>
                </div>
            </div>
        </div>
        )
    }
}
function Card(props){
    const anim = useSpring({
        to:{transform: !props.triggered ?'scale(0.9)':'scale(1.1)',margin:!props.triggered ?'0% 0.5%':'0% 1%'}
    })
    var btn = props.triggered?<button type="button" className={props.btnclass} onClick={()=>props.selectPage(props.src,props.cardClass)}>Select</button>:"";
    return(
        <animated.div onClick = {()=>props.click(props.value)}className="card" style={anim}>
            <img src={props.src} alt=""/>
            <div className="text">
                {btn }
            </div>
        </animated.div>
    )
}
function Body(){
    return(
        <div className="body">
            <div className="label">
                Select A Card Design
            </div>
            <CardDisplay/>
        </div>
    )
}
function About(){
    var sonnetArr=["Let me not to the marriage of true minds\nAdmit impediments. Love is not love\nWhich alters when it alteration finds,\nOr bends with the remover to remove.\nO no! it is an ever-fixed mark\nThat looks on tempests and is never shaken;\nIt is the star to every wand'ring bark,\nWhose worth's unknown, although his height be taken.\nLove's not Time's fool, though rosy lips and cheeks\nWithin his bending sickle's compass come;\nLove alters not with his brief hours and weeks,\nBut bears it out even to the edge of doom.\nIf this be error and upon me prov'd,\nI never writ, nor no man ever lov'd."
    ,"To me, fair friend, you never can be old,\nFor as you were when first your eye I eyed,\nSuch seems your beauty still. Three winters cold\nHave from the forests shook three summers’ pride,\nThree beauteous springs to yellow autumn turned\nIn process of the seasons have I seen,\nThree April perfumes in three hot Junes burned,\nSince first I saw you fresh, which yet are green.\nAh, yet doth beauty, like a dial-hand,\nSteal from his figure, and no pace perceived;\nSo your sweet hue, which methinks still doth stand,\nHath motion, and mine eye may be deceived:\nFor fear of which, hear this, thou age unbred:\nEre you were born was beauty’s summer dead.",
    "Shall I compare thee to a summer’s day?\nThou art more lovely and more temperate:\nRough winds do shake the darling buds of May,\nAnd summer’s lease hath all too short a date;\nSometime too hot the eye of heaven shines,\nAnd often is his gold complexion dimm'd;\nAnd every fair from fair sometime declines,\nBy chance or nature’s changing course untrimm'd;\nBut thy eternal summer shall not fade,\nNor lose possession of that fair thou ow’st;\nNor shall death brag thou wander’st in his shade,\nWhen in eternal lines to time thou grow’st:\n   So long as men can breathe or eyes can see,\n   So long lives this, and this gives life to thee."];
    const [value, setValue] = useState(0);
    const anim = useSpring({
        from:{opacity:0,fontSize:'0.8em'},
        to:{opacity:1,fontSize:'1.2em'},
        reset:true
    })
    function createSonnetDiv(str){
        var sonnet_string = str.split('\n');
        var parentDiv = new Array(14);
        for (let i = 0; i < sonnet_string.length; i++) {
            parentDiv[i] = <div className="sonnet-line">
                {sonnet_string[i]}
            </div>            
        }
        return parentDiv;
    }
    return(
        <div className="abtpage">
            <div className="abtheader">
                About Us
            </div>
            <div className="story">
                <div className="story header">
                    Abstract Of Project
                </div>
                <div className="story-cnt">
                William Shakespeare is often considered as the most prolific author in the history of English. Plays like Hamlet, Macbeth have become topics of research among those who learn literature. One of the greatest works of Shakespeare is the 154 sonnets that he had written which follow a particular and recognized format. Trying to replicate this is a tedious task but that’s what we decided to. Using a char-RNN developed in Pytorch with LSTM Architecture, with the Shakespearean sonnets as input to this network. 
Using this as backend we decided to make it easier for regular people to connect, by using these to generate poems in gift/greeting cards. Gift cards because it is something very personal and lovely. Our hope is to educate people about the great works of literature while promoting love and togetherness in tough pandemic times.
                </div>
            </div>
            <div className="sonnets">
                <div className="sonnet-hdr header">
                    Sonnets
                </div>
                <div className="sonnet cnt">
                    <div className="prev cntr" onClick={() => setValue((value === 0 ? 2 : value-1))}>
                    <img src={require('./Images/previous.svg')} alt=""/>
                    </div>
                    <animated.div style={anim}>
                        {createSonnetDiv(sonnetArr[value])}
                    </animated.div>
                <div className="next cntr" onClick={() => setValue((value + 1)%3)}>
                        <img src={require('./Images/next.svg')} alt=""/>
                    </div>
                </div>
            </div>
        </div>
    )
}
class Page extends React.Component{
    constructor(props){
        super();
        this.state={
            window:"def"
        }
        this.setWindow = this.setWindow.bind(this);
        this.window = <Body/>
    }
    setWindow(flag){
        if(flag){
            this.setState({window:"abt"});
            this.window=<About/>
        }else{
            this.setState({window:"def"});
            this.window=<Body/>
        }
    }
    render(){
        return(
            <div className="Page">
                <Navbar click={this.setWindow}/>
                {this.window}
            </div>
        )
    }
}
ReactDOM.render(
    <Page/>,
  document.getElementById('root')
);
