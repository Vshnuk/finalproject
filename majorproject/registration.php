<?php
    session_start();
    $_SESSION['login_successful']=$_SESSION['login_successful']?$_SESSION['login_successful']:false;
    if($_SESSION['login_successful']==true)
    {
        header("Location: login.html");
    }
    else if(isset($_POST['registration_submit']))
    {
        $con=mysqli_connect("localhost","admin","admin4321","project");
        if(!$con)
            echo ("unsuccesful connection".mysqli_error($con));
        $username=$_POST['username'];
        $password=$_POST['password'];
        $_SESSION['login_successful']=false;
        $check_login="insert into users_login(username,password) values ('$username','$password')";
        $execute_check_login=mysqli_query($con,$check_login);
        if($execute_check_login)
        {
            header("Location: login.html");
        }
        else
        {
            $_SESSION['login_successful']=false;
            echo "<script type='text/javascript'>alert('Registration Failed. Try Registering again!!');location=\"registration.php\"</script>";
        }


    }

?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>
    <link href="//maxcdn.bootstrapcdn.com/bootstrap/3.3.0/css/bootstrap.min.css" rel="stylesheet" id="bootstrap-css">
    <script src="//maxcdn.bootstrapcdn.com/bootstrap/3.3.0/js/bootstrap.min.js"></script>
    <script src="//code.jquery.com/jquery-1.11.1.min.js"></script>
    <link href="registration.css" rel="stylesheet">
</head>
<body>
    <br><br><br><br>
    <div class="container">
		<div class="row">
			<div class="panel panel-primary">
				<div class="panel-body">
					<form method="POST" action="" role="form">
						<div class="form-group">
							<h2>Create account</h2>
						</div>
						<div class="form-group">
							<label class="control-label" for="signupName">Your name</label>
							<input id="signupName" type="text" maxlength="50" class="form-control" name="username">
						</div>
						<div class="form-group">
							<label class="control-label" for="signupPassword">Password</label>
							<input id="signupPassword" type="password" maxlength="25" class="form-control" placeholder="at least 6 characters" length="40" name="password">
						</div>
						<div class="form-group">
							<button id="signupSubmit" type="submit" class="btn btn-info btn-block" name="registration_submit">Create your account</button>
						</div>
						<hr>
						<p></p>Already have an account? <a href="login.html">Sign in</a></p>
					</form>
				</div>
			</div>
		</div>
	</div>

</body>
</html>