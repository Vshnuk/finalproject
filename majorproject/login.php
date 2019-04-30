<?php

    session_start();
    if(isset($_POST['submit']))
    {
        $con=mysqli_connect("localhost","admin","admin4321","project");
        if(!$con)
            echo ("unsuccesful connection".mysqli_error($con));
        $username=$_POST['username'];
        $password=$_POST['password'];
        $_SESSION['login_successful']=true;
        $check_login="select * from users_login where username='$username' and password='$password'";
        $execute_check_login=mysqli_query($con,$check_login);
        $count=0;
        while($row=mysqli_fetch_assoc($execute_check_login))
            $count++;
        if($count==1)
        {
            header("Location: user.php");
        }
        else
        {
            $_SESSION['login_successful']=false;
            echo "<script type='text/javascript'>alert('Login unsuccessful. Try Logging again!!');location=\"login.html\"</script>";
        }
    }
    else
        header("Location: login.html");

?>